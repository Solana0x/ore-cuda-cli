#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "program.h"
#include "unreachable.h"
#include "siphash_rng.h"

#define TARGET_CYCLE 192
#define REQUIREMENT_SIZE 512
#define REQUIREMENT_MUL_COUNT 192
#define REQUIREMENT_LATENCY 195
#define REGISTER_NEEDS_DISPLACEMENT 5
#define PORT_MAP_SIZE (TARGET_CYCLE + 4)
#define NUM_PORTS 3
#define MAX_RETRIES 1
#define LOG2_BRANCH_PROB 4
#define BRANCH_MASK 0x80000000

#define TRACE false
#define TRACE_PRINT(...) do { if (TRACE) printf(__VA_ARGS__); } while (false)

#define MAX(a,b) ((a) > (b) ? (a) : (b))

// Simplified memory access optimization
__device__ bool is_mul(instr_type type) {
    return type <= INSTR_MUL_R;
}

typedef enum execution_port {
    PORT_NONE = 0,
    PORT_P0 = 1,
    PORT_P1 = 2,
    PORT_P5 = 4,
    PORT_P01 = PORT_P0 | PORT_P1,
    PORT_P05 = PORT_P0 | PORT_P5,
    PORT_P015 = PORT_P0 | PORT_P1 | PORT_P5
} execution_port;

typedef struct instr_template {
    instr_type type;
    const char* x86_asm;
    int x86_size;
    int latency;
    execution_port uop1;
    execution_port uop2;
    uint32_t immediate_mask;
    instr_type group;
    bool imm_can_be_0;
    bool distinct_dst;
    bool op_par_src;
    bool has_src;
    bool has_dst;
} instr_template;

typedef struct register_info {
    int latency;
    instr_type last_op;
    int last_op_par;
} register_info;

typedef struct program_item {
    const instr_template** templates;
    uint32_t mask0;
    uint32_t mask1;
    bool duplicates;
} program_item;

typedef struct generator_ctx {
    int cycle;
    int sub_cycle;
    int mul_count;
    bool chain_mul;
    int latency;
    siphash_rng gen;
    register_info registers[8];
    execution_port ports[PORT_MAP_SIZE][NUM_PORTS];
} generator_ctx;

// Load templates and other data into constant memory to reduce memory latency
__constant__ instr_template tpl_umulh_r = { ... };
__constant__ instr_template tpl_smulh_r = { ... };
__constant__ instr_template tpl_mul_r = { ... };
__constant__ instr_template tpl_sub_r = { ... };
__constant__ instr_template tpl_xor_r = { ... };
__constant__ instr_template tpl_add_rs = { ... };
__constant__ instr_template tpl_ror_c = { ... };
__constant__ instr_template tpl_add_c = { ... };
__constant__ instr_template tpl_xor_c = { ... };
__constant__ instr_template tpl_target = { ... };
__constant__ instr_template tpl_branch = { ... };

// Other templates omitted for brevity

__device__ int select_register(int available_regs[8], int regs_count, siphash_rng* gen, int* reg_out) {
    if (regs_count == 0)
        return false;

    int index = regs_count > 1 ? hashx_siphash_rng_u32(gen) % regs_count : 0;
    *reg_out = available_regs[index];
    return true;
}

__device__ int schedule_uop(execution_port uop, generator_ctx* ctx, int cycle, bool commit) {
    for (; cycle < PORT_MAP_SIZE; ++cycle) {
        if ((uop & PORT_P5) && !ctx->ports[cycle][2]) {
            if (commit) ctx->ports[cycle][2] = uop;
            return cycle;
        }
        if ((uop & PORT_P0) && !ctx->ports[cycle][0]) {
            if (commit) ctx->ports[cycle][0] = uop;
            return cycle;
        }
        if ((uop & PORT_P1) && !ctx->ports[cycle][1]) {
            if (commit) ctx->ports[cycle][1] = uop;
            return cycle;
        }
    }
    return -1;
}

__global__ void hashx_program_generate_kernel(const siphash_state* key, hashx_program* program) {
    generator_ctx ctx = {
        .cycle = 0,
        .sub_cycle = 0,
        .mul_count = 0,
        .chain_mul = false,
        .latency = 0,
        .gen = {}, // Explicitly initializing siphash_rng
        .registers = {}, // Explicitly initializing the registers array
        .ports = {} // Explicitly initializing the ports array
    };
    hashx_siphash_rng_init(&ctx.gen, key);
    for (int i = 0; i < 8; ++i) {
        ctx.registers[i].last_op = (instr_type)-1;
        ctx.registers[i].latency = 0;
        ctx.registers[i].last_op_par = -1;
    }
    program->code_size = 0;

    int attempt = 0;
    instr_type last_instr = (instr_type)-1;

    while (program->code_size < HASHX_PROGRAM_MAX_SIZE) {
        instruction* instr = &program->code[program->code_size];

        const instr_template* tpl = select_template(&ctx, last_instr, attempt);
        last_instr = tpl->group;

        instr_from_template(tpl, &ctx.gen, instr);

        int scheduleCycle = schedule_instr(tpl, &ctx, false);
        if (scheduleCycle < 0) break;

        ctx.chain_mul = attempt > 0;

        if (tpl->has_src) {
            if (!select_source(tpl, instr, &ctx, scheduleCycle)) {
                if (attempt++ < MAX_RETRIES) continue;
                ctx.sub_cycle += 3;
                ctx.cycle = ctx.sub_cycle / 3;
                attempt = 0;
                continue;
            }
        }

        if (tpl->has_dst) {
            if (!select_destination(tpl, instr, &ctx, scheduleCycle)) {
                if (attempt++ < MAX_RETRIES) continue;
                ctx.sub_cycle += 3;
                ctx.cycle = ctx.sub_cycle / 3;
                attempt = 0;
                continue;
            }
        }
        attempt = 0;

        scheduleCycle = schedule_instr(tpl, &ctx, true);
        if (scheduleCycle < 0) break;
        if (scheduleCycle >= TARGET_CYCLE) break;

        if (tpl->has_dst) {
            register_info* ri = &ctx.registers[instr->dst];
            int retireCycle = scheduleCycle + tpl->latency;
            ri->latency = retireCycle;
            ri->last_op = tpl->group;
            ri->last_op_par = instr->op_par;
            ctx.latency = MAX(retireCycle, ctx.latency);
        }

        program->code_size++;
        ctx.mul_count += is_mul(instr->opcode);
        ++ctx.sub_cycle;
        ctx.sub_cycle += (tpl->uop2 != PORT_NONE);
        ctx.cycle = ctx.sub_cycle / 3;
    }
    // The check at the end could be offloaded to another kernel if needed for further parallelization
    return
        (program->code_size == REQUIREMENT_SIZE) &
        (ctx.mul_count == REQUIREMENT_MUL_COUNT) &
        (ctx.latency == REQUIREMENT_LATENCY - 1);
}

__host__ bool hashx_program_generate(const siphash_state* key, hashx_program* program) {
    // Call kernel with appropriate grid and block size, tailored to your needs
    hashx_program_generate_kernel<<<1, 1>>>(key, program);
    cudaDeviceSynchronize(); // Ensure kernel execution is complete before returning
}
