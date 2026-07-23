//! b-bit MinHash: minwise signatures truncated to a few bits.

use core::convert::Infallible;
use core::hash::{BuildHasher, Hash};

use alloc::boxed::Box;
use alloc::vec;

use crate::block::PackedArray;
use crate::error::MergeError;
use crate::hash::{DefaultBuildHasher, hash_one, mix64};
use crate::traits::{Insert, Merge, Sketch};

/// A b-bit MinHash signature for approximate Jaccard similarity.
///
/// The working form is identical to [`crate::sketch::MinHash`]: `k` full
/// 64-bit minima, which merge exactly. The b-bit form enters at query
/// time: comparing two signatures uses only the lowest `B` bits of each
/// minimum. Positions then agree either because the true minima match
/// (probability `J`) or because two different minima collide in `B` bits
/// (probability about `2^-B`), and the estimator corrects for the
/// collision term:
///
/// ```text
/// r = fraction of positions whose lowest B bits agree
/// J ~= (r - 2^-B) / (1 - 2^-B)
/// ```
///
/// The paper's point is storage: the materialized signature
/// ([`Self::signature`]) costs `k * B` bits — 8x smaller than full-width
/// at `B = 8` — with only slightly higher estimator variance. Keeping the
/// full minima internally means insertion and merge stay exact; the
/// truncation boundary matters only for the final comparison.
///
/// # References
///
/// - Ping Li and Arnd Christian König, "b-Bit Minwise Hashing", WWW 2010.
///   <https://doi.org/10.1145/1772690.1772759>
/// - Andrei Z. Broder, "On the resemblance and containment of documents",
///   Compression and Complexity of Sequences 1997.
///   <https://doi.org/10.1109/SEQUEN.1997.666900>
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BBitMinHash<const B: u32 = 8, S = DefaultBuildHasher> {
    minima: Box<[u64]>,
    seed_fingerprint: u64,
    hasher: S,
}

impl BBitMinHash<8, DefaultBuildHasher> {
    /// Creates a b-bit MinHash signature of `num_hashes` minima with seed
    /// zero and `B = 8` bits per minimum at query time.
    ///
    /// # Panics
    ///
    /// Panics if `num_hashes` is zero.
    pub fn new(num_hashes: usize) -> Self {
        Self::with_seed(num_hashes, 0)
    }

    /// Creates a b-bit MinHash signature with an explicit hash seed.
    ///
    /// # Panics
    ///
    /// Panics if `num_hashes` is zero.
    pub fn with_seed(num_hashes: usize, seed: u64) -> Self {
        let hasher = DefaultBuildHasher::new(seed);
        Self::from_parts(num_hashes, hasher.seed_fingerprint(), hasher)
    }
}

impl<const B: u32, S> BBitMinHash<B, S> {
    /// Creates a b-bit MinHash sketch from explicit components.
    ///
    /// # Panics
    ///
    /// Panics if `num_hashes` is zero.
    pub fn from_parts(num_hashes: usize, seed_fingerprint: u64, hasher: S) -> Self {
        assert!(num_hashes > 0, "b-bit MinHash needs at least one hash");
        Self {
            minima: vec![u64::MAX; num_hashes].into_boxed_slice(),
            seed_fingerprint,
            hasher,
        }
    }

    /// Returns the number of minima kept per signature.
    pub fn num_hashes(&self) -> usize {
        self.minima.len()
    }

    /// Returns the seed fingerprint used by merge compatibility checks.
    pub const fn seed_fingerprint(&self) -> u64 {
        self.seed_fingerprint
    }

    /// Returns the full-width minimum at position `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.num_hashes()`.
    pub fn minimum(&self, index: usize) -> u64 {
        self.minima[index]
    }

    /// Returns the truncated `B`-bit minimum at position `index`, the value
    /// comparisons are made on.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.num_hashes()`.
    pub fn truncated_minimum(&self, index: usize) -> u64 {
        self.minima[index] & PackedArray::<B>::MAX
    }

    /// Materializes the compact `B`-bit signature: `k * B` bits, suitable
    /// for storage or transmission.
    pub fn signature(&self) -> PackedArray<B> {
        let mut signature = PackedArray::new(self.num_hashes());
        for (i, &minimum) in self.minima.iter().enumerate() {
            signature.set(i, minimum & PackedArray::<B>::MAX);
        }
        signature
    }

    /// Returns the byte length of the full-minima working storage. The
    /// materialized signature is `k * B / 8` bytes.
    pub fn storage_bytes(&self) -> usize {
        self.minima.len() * size_of::<u64>()
    }

    /// Clears the signature back to all-maxima.
    pub fn clear(&mut self) {
        self.minima.fill(u64::MAX);
    }

    /// Checks that two signatures can be compared or merged.
    ///
    /// # Panics
    ///
    /// Panics if the signatures have different lengths or seeds.
    fn check_compatible(&self, other: &Self) {
        assert_eq!(
            self.num_hashes(),
            other.num_hashes(),
            "b-bit MinHash signatures must have the same number of hashes"
        );
        assert_eq!(
            self.seed_fingerprint, other.seed_fingerprint,
            "b-bit MinHash signatures must share a hash seed"
        );
    }
}

impl<const B: u32, S> BBitMinHash<B, S>
where
    S: BuildHasher,
{
    /// Inserts `item`, lowering each full minimum whose derived value is
    /// smaller.
    pub fn insert_item<T>(&mut self, item: &T)
    where
        T: Hash + ?Sized,
    {
        let hash = hash_one(&self.hasher, item);
        for (i, slot) in self.minima.iter_mut().enumerate() {
            let derived = mix64(hash ^ (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
            *slot = (*slot).min(derived);
        }
    }

    /// Estimates the Jaccard similarity with `other` from the fraction of
    /// positions whose truncated minima agree, corrected for `B`-bit
    /// collisions.
    ///
    /// # Panics
    ///
    /// Panics if the two sketches use different hash counts or seeds.
    pub fn jaccard(&self, other: &Self) -> f64 {
        self.check_compatible(other);
        let equal = (0..self.num_hashes())
            .filter(|&i| self.truncated_minimum(i) == other.truncated_minimum(i))
            .count();
        let r = equal as f64 / self.num_hashes() as f64;
        let collision = 1.0 / (1_u64 << B.min(63)) as f64;
        ((r - collision) / (1.0 - collision)).clamp(0.0, 1.0)
    }
}

impl<const B: u32, S> Sketch for BBitMinHash<B, S> {
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

impl<T, const B: u32, S> Insert<T> for BBitMinHash<B, S>
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

impl<const B: u32, S> Merge for BBitMinHash<B, S> {
    /// Merges by element-wise minimum of the *full* minima: exactly the
    /// working form of the union, like [`crate::sketch::MinHash`]. The
    /// truncation only applies to later comparisons.
    fn merge_from(&mut self, other: &Self) -> Result<(), MergeError> {
        if self.num_hashes() != other.num_hashes() {
            return Err(MergeError::GeometryMismatch);
        }
        if self.seed_fingerprint != other.seed_fingerprint {
            return Err(MergeError::SeedMismatch);
        }
        for (left, right) in self.minima.iter_mut().zip(other.minima.iter()) {
            *left = (*left).min(*right);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::BBitMinHash;
    use crate::error::MergeError;
    use crate::traits::{Insert, Merge};

    #[test]
    fn identical_sets_have_similarity_one() {
        let mut left = BBitMinHash::new(128);
        let mut right = BBitMinHash::new(128);
        for i in 0..1_000_u64 {
            left.insert_item(&i);
            right.insert_item(&i);
        }
        assert_eq!(left.jaccard(&right), 1.0);
    }

    #[test]
    fn disjoint_sets_have_near_zero_similarity() {
        let mut left = BBitMinHash::new(128);
        let mut right = BBitMinHash::new(128);
        for i in 0..1_000_u64 {
            left.insert_item(&i);
            right.insert_item(&(1_000_000 + i));
        }
        // Collision noise means near-zero, not exactly zero.
        assert!(left.jaccard(&right) < 0.05);
    }

    #[test]
    fn signature_is_one_byte_per_minimum() {
        let sketch = BBitMinHash::<8>::new(128);
        assert_eq!(sketch.signature().storage_bytes(), 128);
        assert_eq!(sketch.storage_bytes(), 128 * size_of::<u64>());
    }

    #[test]
    fn merge_is_exact_on_full_minima_and_validates() {
        let mut left = BBitMinHash::with_seed(64, 1);
        let mut right = BBitMinHash::with_seed(64, 1);
        left.insert_item(&1_u64);
        right.insert_item(&2_u64);

        let before: Vec<u64> = (0..64).map(|i| left.minimum(i)).collect();
        left.merge_from(&right).unwrap();
        for (i, &previous) in before.iter().enumerate() {
            assert_eq!(left.minimum(i), previous.min(right.minimum(i)));
        }

        let other_k = BBitMinHash::with_seed(32, 1);
        assert_eq!(left.merge_from(&other_k), Err(MergeError::GeometryMismatch));
        let other_seed = BBitMinHash::with_seed(64, 2);
        assert_eq!(left.merge_from(&other_seed), Err(MergeError::SeedMismatch));
    }

    #[test]
    fn capability_traits_work() {
        let mut sketch = BBitMinHash::new(64);
        Insert::<str>::insert(&mut sketch, "alice").unwrap();
        assert!((0..64).any(|i| sketch.minimum(i) != u64::MAX));
    }
}
