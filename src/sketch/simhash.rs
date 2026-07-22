//! SimHash implementation for cosine similarity.

use core::convert::Infallible;
use core::hash::{BuildHasher, Hash};

use alloc::boxed::Box;

use crate::error::MergeError;
use crate::hash::{DefaultBuildHasher, hash_one};
use crate::traits::{Insert, Merge, Sketch};

/// The signature width in bits.
pub const SIMHASH_BITS: usize = 64;

/// A SimHash signature for approximate cosine similarity between item
/// multisets (seen as 0/1 incidence vectors).
///
/// Every item is hashed to a 64-bit value; the sketch keeps a per-bit
/// running sum of `+1` for set bits and `-1` for clear bits. The signature
/// is the sign pattern of the sums. Two sets' signatures differ in a
/// fraction of bits proportional to the angle between their incidence
/// vectors, so Hamming distance estimates the cosine:
///
/// ```text
/// insert("x"): for each bit j of hash("x"): sum[j] += (bit ? +1 : -1)
///
/// signature = [sum[j] >= 0 for j in 0..64]
///
/// angle(A, B) ~= pi * hamming(A.sig ^ B.sig) / 64
/// cosine(A, B) ~= cos(angle)
/// ```
///
/// The sketch's key operational property: it is *linear*. Merging two
/// sketches is an element-wise sum of the sums, giving exactly the sketch
/// of the multiset union — the trick behind its use for near-duplicate
/// detection at scale.
///
/// # References
///
/// - Moses S. Charikar, "Similarity Estimation Techniques from Rounding
///   Algorithms", STOC 2002. <https://doi.org/10.1145/509907.509965>
/// - Gurmeet Singh Manku, Arvind Jain, and Anish Das Sarma, "Detecting
///   Near-Duplicates for Web Crawling", WWW 2007.
///   <https://doi.org/10.1145/1242572.1242592>
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SimHash<S = DefaultBuildHasher> {
    sums: Box<[i64]>,
    seed_fingerprint: u64,
    hasher: S,
}

impl SimHash<DefaultBuildHasher> {
    /// Creates a SimHash sketch with hash seed zero.
    pub fn new() -> Self {
        Self::with_seed(0)
    }

    /// Creates a SimHash sketch with an explicit hash seed.
    pub fn with_seed(seed: u64) -> Self {
        let hasher = DefaultBuildHasher::new(seed);
        Self::from_parts(hasher.seed_fingerprint(), hasher)
    }
}

impl Default for SimHash<DefaultBuildHasher> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S> SimHash<S> {
    /// Creates a SimHash sketch from explicit components.
    pub fn from_parts(seed_fingerprint: u64, hasher: S) -> Self {
        Self {
            sums: alloc::vec![0_i64; SIMHASH_BITS].into_boxed_slice(),
            seed_fingerprint,
            hasher,
        }
    }

    /// Returns the seed fingerprint used by merge compatibility checks.
    pub const fn seed_fingerprint(&self) -> u64 {
        self.seed_fingerprint
    }

    /// Returns the current per-bit sums.
    pub fn sums(&self) -> &[i64] {
        &self.sums
    }

    /// Returns the 64-bit signature: bit `j` is set iff `sum[j] >= 0`.
    pub fn signature(&self) -> u64 {
        let mut signature = 0_u64;
        for (j, &sum) in self.sums.iter().enumerate() {
            if sum >= 0 {
                signature |= 1 << j;
            }
        }
        signature
    }

    /// Returns the byte length of the sum storage.
    pub fn storage_bytes(&self) -> usize {
        SIMHASH_BITS * size_of::<i64>()
    }

    /// Clears all sums.
    pub fn clear(&mut self) {
        self.sums.fill(0);
    }

    /// Checks that two sketches share a seed.
    ///
    /// # Panics
    ///
    /// Panics if the seeds differ.
    fn check_compatible(&self, other: &Self) {
        assert_eq!(
            self.seed_fingerprint, other.seed_fingerprint,
            "SimHash sketches must share a hash seed"
        );
    }
}

impl<S> SimHash<S>
where
    S: BuildHasher,
{
    /// Inserts `item`, adding `+1`/`-1` per hash bit to the sums.
    pub fn insert_item<T>(&mut self, item: &T)
    where
        T: Hash + ?Sized,
    {
        let hash = hash_one(&self.hasher, item);
        for (j, sum) in self.sums.iter_mut().enumerate() {
            if hash >> j & 1 == 1 {
                *sum += 1;
            } else {
                *sum -= 1;
            }
        }
    }

    /// Returns the Hamming distance between the two signatures.
    ///
    /// # Panics
    ///
    /// Panics if the sketches use different hash seeds.
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        self.check_compatible(other);
        (self.signature() ^ other.signature()).count_ones()
    }

    /// Estimates the cosine similarity of the two item multisets as
    /// `cos(pi * hamming / 64)`.
    ///
    /// Available with the `std` or `libm` feature.
    ///
    /// # Panics
    ///
    /// Panics if the sketches use different hash seeds.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn estimated_cosine(&self, other: &Self) -> f64 {
        let angle =
            core::f64::consts::PI * f64::from(self.hamming_distance(other)) / SIMHASH_BITS as f64;
        crate::float::cos(angle)
    }
}

impl<S> Sketch for SimHash<S> {
    fn clear(&mut self) {
        self.clear();
    }

    fn len_hint(&self) -> Option<u64> {
        None
    }

    fn storage_bytes(&self) -> usize {
        self.storage_bytes()
    }
}

impl<T, S> Insert<T> for SimHash<S>
where
    T: Hash + ?Sized,
    S: BuildHasher,
{
    type Err = Infallible;

    fn insert(&mut self, item: &T) -> Result<(), Self::Err> {
        self.insert_item(item);
        Ok(())
    }
}

impl<S> Merge for SimHash<S> {
    /// Merges by element-wise sum of the sums: the result is exactly the
    /// sketch of the multiset union.
    fn merge_from(&mut self, other: &Self) -> Result<(), MergeError> {
        if self.seed_fingerprint != other.seed_fingerprint {
            return Err(MergeError::SeedMismatch);
        }
        for (left, right) in self.sums.iter_mut().zip(other.sums.iter()) {
            *left += right;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::SimHash;
    use crate::error::MergeError;
    use crate::traits::{Insert, Merge};

    #[test]
    fn identical_sets_have_zero_distance() {
        let mut left = SimHash::new();
        let mut right = SimHash::new();
        for i in 0..1_000_u64 {
            left.insert_item(&i);
            right.insert_item(&i);
        }
        assert_eq!(left.hamming_distance(&right), 0);
        assert_eq!(left.estimated_cosine(&right), 1.0);
    }

    #[test]
    fn very_different_sets_have_large_distance() {
        // Two disjoint large sets behave like independent vectors: the
        // signatures should differ in roughly half the bits (cosine ~0).
        let mut left = SimHash::new();
        let mut right = SimHash::new();
        for i in 0..10_000_u64 {
            left.insert_item(&i);
            right.insert_item(&(1_000_000 + i));
        }
        let distance = left.hamming_distance(&right);
        assert!(distance > 16, "distance {distance} suspiciously small");
    }

    #[test]
    fn merge_is_exactly_the_union_sketch() {
        let mut left = SimHash::with_seed(7);
        let mut right = SimHash::with_seed(7);
        let mut single = SimHash::with_seed(7);
        for i in 0..1_000_u64 {
            left.insert_item(&i);
            single.insert_item(&i);
        }
        for i in 1_000..2_000_u64 {
            right.insert_item(&i);
            single.insert_item(&i);
        }
        left.merge_from(&right).unwrap();
        assert_eq!(left.sums(), single.sums());

        let other_seed = SimHash::with_seed(8);
        assert_eq!(left.merge_from(&other_seed), Err(MergeError::SeedMismatch));
    }

    #[test]
    fn capability_traits_work() {
        let mut sketch = SimHash::new();
        Insert::<str>::insert(&mut sketch, "alice").unwrap();
        assert!(sketch.sums().iter().any(|&s| s != 0));
    }
}
