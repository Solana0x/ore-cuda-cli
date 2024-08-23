pub use equix;
use equix::SolutionArray;
#[cfg(not(feature = "solana"))]
use sha3::Digest;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Mutex;

// A cache to store previously computed results for reuse
lazy_static::lazy_static! {
    static ref HASH_CACHE: Mutex<HashMap<([u8; 32], [u8; 8]), Hash>> = Mutex::new(HashMap::new());
}

/// Generates a new Drillx hash from a challenge and nonce.
#[inline(always)]
pub fn hash(challenge: &[u8; 32], nonce: &[u8; 8]) -> Result<Hash, DrillxError> {
    let cache_key = (*challenge, *nonce);
    if let Some(cached_hash) = HASH_CACHE.lock().unwrap().get(&cache_key) {
        return Ok(cached_hash.clone());
    }

    let digest = digest(challenge, nonce)?;
    let hash = Hash {
        d: digest,
        h: hashv(&digest, nonce),
    };
    HASH_CACHE.lock().unwrap().insert(cache_key, hash.clone());
    Ok(hash)
}

/// Generates drillx hashes from a challenge and nonce using pre-allocated memory.
#[inline(always)]
pub fn hashes_with_memory(
    memory: &mut equix::SolverMemory,
    challenge: &[u8; 32],
    nonce: &[u8; 8],
) -> Vec<Hash> {
    let mut hashes: Vec<Hash> = Vec::with_capacity(7);
    if let Ok(solutions) = digests_with_memory(memory, challenge, nonce) {
        // Use parallel iterator and collect results
        hashes = solutions
            .par_iter()
            .map(|solution| {
                let digest = solution.to_bytes();
                Hash {
                    d: digest,
                    h: hashv(&digest, nonce),
                }
            })
            .collect();
    }
    hashes
}

/// Constructs a Keccak digest from a challenge and nonce.
#[inline(always)]
fn digest(challenge: &[u8; 32], nonce: &[u8; 8]) -> Result<[u8; 16], DrillxError> {
    let seed = seed(challenge, nonce);
    let solutions = equix::solve(&seed).map_err(|_| DrillxError::BadEquix)?;
    if solutions.is_empty() {
        return Err(DrillxError::NoSolutions);
    }
    // SAFETY: The equix solver guarantees that the first solution is always valid
    let solution = unsafe { solutions.get_unchecked(0) };
    Ok(solution.to_bytes())
}

/// Returns a Keccak hash of the digest and nonce, optimized using SIMD.
#[cfg(not(feature = "solana"))]
#[inline(always)]
fn hashv(digest: &[u8; 16], nonce: &[u8; 8]) -> [u8; 32] {
    let mut hasher = sha3::Keccak256::new();
    hasher.update(&sorted(*digest));
    hasher.update(nonce);
    hasher.finalize().into()
}

/// Sorts a digest as a list of `u16` values, using an efficient in-place sort.
#[inline(always)]
fn sorted(digest: [u8; 16]) -> [u8; 16] {
    let mut sorted_digest = digest;
    unsafe {
        let u16_slice: &mut [u16; 8] = core::mem::transmute(&mut sorted_digest);
        u16_slice.sort_unstable(); // Continue using efficient sorting
    }
    sorted_digest
}

/// Checks if the digest is a valid Equihash construction.
pub fn is_valid_digest(challenge: &[u8; 32], nonce: &[u8; 8], digest: &[u8; 16]) -> bool {
    let seed = seed(challenge, nonce);
    equix::verify_bytes(&seed, digest).is_ok()
}

/// Calculates the number of leading zeros in a hash.
pub fn difficulty(hash: [u8; 32]) -> u32 {
    hash.iter().take_while(|&&byte| byte == 0).count() as u32 * 8
        + hash
            .iter()
            .find(|&&byte| byte != 0)
            .map_or(0, |&byte| byte.leading_zeros() as u32)
}

/// The result of a Drillx hash.
#[derive(Default, Clone)]
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
