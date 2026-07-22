//! Deterministic item hashing.

use core::hash::{BuildHasher, Hash, Hasher};

use super::mix64;

/// A deterministic, seedable [`BuildHasher`] used by default.
///
/// This hasher is intentionally small and stable. It is not a cryptographic
/// hash and should not be used as a hash-flooding defense.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DefaultBuildHasher {
    seed: u64,
}

impl DefaultBuildHasher {
    /// Creates a hasher builder with an explicit seed.
    pub const fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Returns the configured seed.
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Returns a stable fingerprint used for merge compatibility checks.
    pub fn seed_fingerprint(&self) -> u64 {
        mix64(self.seed ^ 0x3d9f_2c68_2a8d_1f31)
    }
}

impl Default for DefaultBuildHasher {
    fn default() -> Self {
        Self::new(0)
    }
}

impl BuildHasher for DefaultBuildHasher {
    type Hasher = StableHasher;

    fn build_hasher(&self) -> Self::Hasher {
        StableHasher::new(self.seed)
    }
}

/// The streaming hasher produced by [`DefaultBuildHasher`].
#[derive(Clone, Debug)]
pub struct StableHasher {
    state: u64,
    len: u64,
}

impl StableHasher {
    /// Creates a streaming hasher with an explicit seed.
    pub fn new(seed: u64) -> Self {
        Self {
            state: mix64(seed ^ 0x243f_6a88_85a3_08d3),
            len: 0,
        }
    }
}

impl Hasher for StableHasher {
    fn finish(&self) -> u64 {
        mix64(self.state ^ self.len.wrapping_mul(0x9e37_79b9_7f4a_7c15))
    }

    fn write(&mut self, bytes: &[u8]) {
        self.len = self
            .len
            .checked_add(bytes.len() as u64)
            .expect("hashed byte length overflowed u64");

        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            let mut lane = [0; 8];
            lane.copy_from_slice(chunk);
            self.write_lane(u64::from_le_bytes(lane));
        }

        let remainder = chunks.remainder();
        if !remainder.is_empty() {
            let mut lane = 0_u64;
            for (i, byte) in remainder.iter().enumerate() {
                lane |= (*byte as u64) << (i * 8);
            }
            lane ^= (remainder.len() as u64) << 56;
            self.write_lane(lane);
        }
    }
}

impl StableHasher {
    fn write_lane(&mut self, lane: u64) {
        self.state ^= mix64(lane.wrapping_add(0x9e37_79b9_7f4a_7c15));
        self.state = self
            .state
            .rotate_left(27)
            .wrapping_mul(0x3c79_ac49_2ba7_b653)
            .wrapping_add(0x1c69_b3f7_4ac4_ae35);
    }
}

/// Hashes application items into the 64-bit hash consumed by index schemes.
pub trait ItemHasher<T: ?Sized> {
    /// Hashes `item`.
    fn hash_item(&self, item: &T) -> u64;
}

impl<T, H> ItemHasher<T> for H
where
    T: Hash + ?Sized,
    H: BuildHasher,
{
    fn hash_item(&self, item: &T) -> u64 {
        hash_one(self, item)
    }
}

/// Hashes one item with a [`BuildHasher`].
pub fn hash_one<T, H>(hasher: &H, item: &T) -> u64
where
    T: Hash + ?Sized,
    H: BuildHasher,
{
    hasher.hash_one(item)
}

#[cfg(test)]
mod tests {
    use super::{DefaultBuildHasher, hash_one};

    #[test]
    fn default_hasher_is_seeded_and_stable() {
        let a = DefaultBuildHasher::new(7);
        let b = DefaultBuildHasher::new(7);
        let c = DefaultBuildHasher::new(8);

        assert_eq!(hash_one(&a, "same item"), hash_one(&b, "same item"));
        assert_ne!(hash_one(&a, "same item"), hash_one(&c, "same item"));
    }
}
