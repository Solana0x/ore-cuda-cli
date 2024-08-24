# ORE CUDA CLI OPTIMISED CODE FOR GPU

``` Optimised for RTX 4090 GPU : 2.2-2KH/s```

# Features 

1. Optimised for RTX 4090 GPU
2. Current Hash rate = 2200-2000 H/s Avg difficulty Hit = 18+
3. Works on CUDA Nvidia tool Kit (v12.5+)

# Installation  

1. Install git repo by `git clone https://github.com/Solana0x/ore-cuda-cli`
2. move inside the cuda directory `cd ore-cuda-cli`
3. To install all the dependencies at once `bash install.sh`
4. Below stpes to install one by one =>
5. upgrade and update by - `sudo apt update && sudo apt upgrade -y`
6. install `sudo apt-get install -y pkg-config libssl-dev build-essential curl`
7. install rust `curl https://sh.rustup.rs -sSf | sh -s -- -y`
8. Add rust to env path `source "$HOME/.cargo/env"`
9. install Solana `sh -c "$(curl -sSfL https://release.solana.com/v1.18.4/install)"`
10. Add Solana to env path `export PATH="$HOME/.local/share/solana/install/active_release/bin:$PATH"`
11. Build the project with Cargo `cargo build --release --features gpu`
12. Add your Solana PRIVATE KEY key.json file by `nano key.json`
13. To run the Code `./target/release/ore --rpc url mine --keypair key.json --priority-fee 10000 `

![image](https://github.com/user-attachments/assets/4450ceb8-84c3-4205-9fb2-0f8a1420c257)

# FOR ANY KIND OF HELP CONTACT : 0xphatom on Discord https://discord.com/users/979641024215416842
