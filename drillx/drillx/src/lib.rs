pub use equix;
use equix::SolutionArray;
#[cfg(not(feature = "solana"))]
use sha3::Digest;
use rayon::prelude::*;
use tokio::task;

/// Generates a new Drillx hash from a challenge and nonce asynchronously.
#[inline(always)]
pub async fn hash_async(challenge: &[u8; 32], nonce: &[u8; 8]) -> Result<Hash, DrillxError> {
    let challenge = *challenge; // Copy the data to avoid borrowing issues
    let nonce = *nonce;         // Copy the data to avoid borrowing issues
    let digest_result = task::spawn_blocking(move || digest(&challenge, &nonce))
        .await
        .map_err(|_| DrillxError::JoinError)?; // Handle JoinError

    let digest = digest_result?; // Handle DrillxError from digest function
    Ok(Hash {
        d: digest,
        h: hashv(&digest, &nonce),
    })
}

/// Generates a new Drillx hash using pre-allocated memory asynchronously.
#[inline(always)]
pub async fn hash_with_memory_async(
    memory: &mut equix::SolverMemory,
    challenge: &[u8; 32],
    nonce: &[u8; 8],
) -> Result<Hash, DrillxError> {
    let challenge = *challenge; // Copy the data to avoid borrowing issues
    let nonce = *nonce;         // Copy the data to avoid borrowing issues
    let memory = memory.clone(); // Clone memory for thread safety

    let digest_result = task::spawn_blocking(move || digest_with_memory(&mut memory.clone(), &challenge, &nonce))
        .await
        .map_err(|_| DrillxError::JoinError)?; // Handle JoinError

    let digest = digest_result?; // Handle DrillxError from digest_with_memory function
    Ok(Hash {
        d: digest,
        h: hashv(&digest, &nonce),
    })
}

// The rest of the code remains unchanged

/// Generates Drillx hashes from a challenge and nonce using pre-allocated memory and parallel processing.
#[inline(always)]
pub fn hashes_with_memory_parallel(
    memory: &mut equix::SolverMemory,
    challenge: &[u8; 32],
    nonce: &[u8; 8],
) -> Vec<Hash> {
    if let Ok(solutions) = digests_with_memory(memory, challenge, nonce) {
        solutions
            .par_iter()
            .map(|solution| {
                let digest = solution.to_bytes();
                Hash {
                    d: digest,
                    h: hashv(&digest, nonce),
                }
            })
            .collect()
    } else {
        Vec::new()
    }
}

/// Concatenates a challenge and nonce into a single buffer.
#[inline(always)]
pub fn seed(challenge: &[u8; 32], nonce: &[u8; 8]) -> [u8; 40] {
    let mut result = [0; 40];
    result[..32].copy_from_slice(challenge);
    result[32..].copy_from_slice(nonce);
    result
}

/// Constructs a Keccak digest from a challenge and nonce.
#[inline(always)]
fn digest(challenge: &[u8; 32], nonce: &[u8; 8]) -> Result<[u8; 16], DrillxError> {
    let seed = seed(challenge, nonce);
    let solutions = equix::solve(&seed).map_err(|_| DrillxError::BadEquix)?;
    solutions.get(0).map_or(Err(DrillxError::NoSolutions), |s| Ok(s.to_bytes()))
}

/// Constructs a Keccak digest using pre-allocated memory.
#[inline(always)]
fn digest_with_memory(
    memory: &mut equix::SolverMemory,
    challenge: &[u8; 32],
    nonce: &[u8; 8],
) -> Result<[u8; 16], DrillxError> {
    let seed = seed(challenge, nonce);
    let equix = equix::EquiXBuilder::new()
        .runtime(equix::RuntimeOption::TryCompile)
        .build(&seed)
        .map_err(|_| DrillxError::BadEquix)?;
    let solutions = equix.solve_with_memory(memory);
    solutions.get(0).map_or(Err(DrillxError::NoSolutions), |s| Ok(s.to_bytes()))
}

/// Constructs a Keccak digest from a challenge and nonce using equix hashes and pre-allocated memory.
#[inline(always)]
fn digests_with_memory(
    memory: &mut equix::SolverMemory,
    challenge: &[u8; 32],
    nonce: &[u8; 8],
) -> Result<SolutionArray, DrillxError> {
    let seed = seed(challenge, nonce);
    let equix = equix::EquiXBuilder::new()
        .runtime(equix::RuntimeOption::TryCompile)
        .build(&seed)
        .map_err(|_| DrillxError::BadEquix)?;
    Ok(equix.solve_with_memory(memory))
}

/// Sorts a digest as a list of `u16` values.
#[inline(always)]
fn sorted(mut digest: [u8; 16]) -> [u8; 16] {
    let u16_slice: &mut [u16; 8] = bytemuck::cast_mut(&mut digest);
    u16_slice.sort_unstable();
    digest
}

/// Returns a Keccak hash of the digest and nonce.
#[cfg(feature = "solana")]
#[inline(always)]
fn hashv(digest: &[u8; 16], nonce: &[u8; 8]) -> [u8; 32] {
    solana_program::keccak::hashv(&[&sorted(*digest), nonce]).to_bytes()
}

/// Calculates a hash using SHA3-256.
#[cfg(not(feature = "solana"))]
#[inline(always)]
fn hashv(digest: &[u8; 16], nonce: &[u8; 8]) -> [u8; 32] {
    let mut hasher = sha3::Keccak256::new();
    hasher.update(&sorted(*digest));
    hasher.update(nonce);
    hasher.finalize().into()
}

/// Checks if the digest is a valid Equihash construction.
pub fn is_valid_digest(challenge: &[u8; 32], nonce: &[u8; 8], digest: &[u8; 16]) -> bool {
    let seed = seed(challenge, nonce);
    equix::verify_bytes(&seed, digest).is_ok()
}

/// Returns the number of leading zeros in a 32-byte buffer.
pub fn difficulty(hash: [u8; 32]) -> u32 {
    hash.iter().fold(0, |acc, &byte| {
        let lz = byte.leading_zeros();
        acc + lz - ((lz < 8) as u32 * (8 - lz))
    })
}

/// The result of a Drillx hash.
#[derive(Default)]
pub struct Hash {
    pub d: [u8; 16], // digest
    pub h: [u8; 32], // hash
}

impl Hash {
    /// The leading number of zeros in the hash.
    pub fn difficulty(&self) -> u32 {
        difficulty(self.h)
    }
}

/// A Drillx solution that can be efficiently validated on-chain.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct Solution {
    pub d: [u8; 16], // digest
    pub n: [u8; 8],  // nonce
}

impl Solution {
    /// Builds a new verifiable solution from a digest and nonce.
    pub fn new(digest: [u8; 16], nonce: [u8; 8]) -> Solution {
        Solution { d: digest, n: nonce }
    }

    /// Checks if the solution is valid.
    pub fn is_valid(&self, challenge: &[u8; 32]) -> bool {
        is_valid_digest(challenge, &self.n, &self.d)
    }

    /// Calculates the result hash for the solution.
    pub fn to_hash(&self) -> Hash {
        Hash {
            d: self.d,
            h: hashv(&self.d, &self.n),
        }
    }
}

#[derive(Debug)]
pub enum DrillxError {
    BadEquix,
    NoSolutions,
    JoinError, // New error type for handling JoinError
}

impl std::fmt::Display for DrillxError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            DrillxError::BadEquix => write!(f, "Failed Equix"),
            DrillxError::NoSolutions => write!(f, "No solutions"),
            DrillxError::JoinError => write!(f, "JoinError in task execution"), // Message for new error type
        }
    }
}

impl std::error::Error for DrillxError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}
