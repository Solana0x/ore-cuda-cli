use bytemuck::{Pod, Zeroable};
use drillx::Solution;
use solana_program::{
    account_info::AccountInfo,
    declare_id,
    entrypoint::ProgramResult,
    instruction::{AccountMeta, Instruction},
    log::sol_log_compute_units,
    program_error::ProgramError,
    pubkey::Pubkey,
};

declare_id!("mineRHF5r6S7HyD9SppBfVMXMavDkJsxwGesEvxZr2A");

#[cfg(not(feature = "no-entrypoint"))]
solana_program::entrypoint!(process_instruction);

pub fn process_instruction(
    _program_id: &Pubkey,
    accounts: &[AccountInfo],
    data: &[u8],
) -> ProgramResult {
    let args = Args::try_from_bytes(data)?;
    if accounts.len() < 1 {
        return Err(ProgramError::NotEnoughAccountKeys);
    }

    // Prove that the solution is valid.
    static CHALLENGE: [u8; 32] = [255; 32];

    let solution = Solution {
        d: args.digest,
        n: args.nonce,
    };

    if !solution.is_valid(&CHALLENGE) {
        return Err(ProgramError::Custom(0));
    }

    if solution.to_hash().difficulty() < args.difficulty as u32 {
        return Err(ProgramError::Custom(1));
    }

    sol_log_compute_units(); // Log the compute units used, useful for debugging and optimization.
    Ok(())
}

pub fn verify(signer: Pubkey, difficulty: u64, nonce: [u8; 8], digest: [u8; 16]) -> Instruction {
    Instruction {
        program_id: crate::id(),
        accounts: vec![AccountMeta::new(signer, true)],
        data: Args {
            difficulty,
            digest,
            nonce,
        }
        .to_bytes(), // Directly convert to bytes slice
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Args {
    pub difficulty: u64,
    pub digest: [u8; 16],
    pub nonce: [u8; 8],
}

impl Args {
    pub fn to_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }

    fn try_from_bytes(data: &[u8]) -> Result<&Self, ProgramError> {
        bytemuck::try_from_bytes(data).map_err(|_| ProgramError::InvalidAccountData)
    }
}
