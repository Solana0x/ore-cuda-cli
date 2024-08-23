pub use equix;
use equix::SolutionArray;
#[cfg(not(feature = "solana"))]
use sha3::Digest;
use rayon::prelude::*;
use std::sync::Mutex;
use smallvec::SmallVec;

/// Generates a new Drillx hash from a challenge and nonce.
#[inline(always)]
pub fn hash(challenge: &[u8; 32], nonce: &[u8; 8]) -> Result<Hash, DrillxError> {
    let digest = digest(challenge, nonce)?;
    Ok(Hash {
        d: digest,
        h: hashv(&digest, nonce),
    })
}

/// Generates a new Drillx hash using pre-allocated memory.
#[inline(always)]
pub fn hash_with_memory(
    memory: &mut equix::SolverMemory,
    challenge: &[u8; 32],
    nonce: &[u8; 8],
) -> Result<Hash, DrillxError> {
    let digest = digest_with_memory(memory, challenge, nonce)?;
    Ok(Hash {
        d: digest,
        h: hashv(&digest, nonce),
    })
}

/// Generates drillx hashes from a challenge and nonce using pre-allocated memory.
#[inline(always)]
pub fn hashes_with_memory(
    memory: &mut equix::SolverMemory,
    challenge: &[u8; 32],
    nonce: &[u8; 8],
) -> Vec<Hash> {
    let hashes = Mutex::new(Vec::with_capacity(7));
    if let Ok(solutions) = digests_with_memory(memory, challenge, nonce) {
        solutions
            .par_iter() 
            .for_each(|solution| {
                let digest = solution.to_bytes();
                let hash = Hash {
                    d: digest,
                    h: hashv(&digest, nonce),
                };
                hashes.lock().unwrap().push(hash);
            });
    }
    hashes.into_inner().unwrap()
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
    if solutions.is_empty() {
        return Err(DrillxError::NoSolutions);
    }
    let solution = unsafe { solutions.get_unchecked(0) };
    Ok(solution.to_bytes())
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
    if solutions.is_empty() {
        return Err(DrillxError::NoSolutions);
    }
    let solution = unsafe { solutions.get_unchecked(0) };
    Ok(solution.to_bytes())
}

/// Constructs a keccak digest from a challenge and nonce using equix hashes and pre-allocated memory.
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
fn sorted(digest: [u8; 16]) -> [u8; 16] {
    let mut sorted_digest = digest;
    unsafe {
        let u16_slice: &mut [u16; 8] = core::mem::transmute(&mut sorted_digest);
        u16_slice.sort_unstable();
    }
    sorted_digest
}

/// Returns a Keccak hash of the digest and nonce.
#[cfg(feature = "solana")]
#[inline(always)]
fn hashv(digest: &[u8; 16], nonce: &[u8; 8]) -> [u8; 32] {
    solana_program::keccak::hashv(&[sorted(*digest).as_slice(), nonce.as_slice()]).to_bytes()
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
    let mut count = 0;
    for &byte in &hash {
        let lz = byte.leading_zeros();
        count += lz;
        if lz < 8 {
            break;
        }
    }
    count
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
}

impl std::fmt::Display for DrillxError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            DrillxError::BadEquix => write!(f, "Failed Equix"),
            DrillxError::NoSolutions => write!(f, "No solutions"),
        }
    }
}

impl std::error::Error for DrillxError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}
