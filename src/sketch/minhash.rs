//! MinHash implementation.

use core::convert::Infallible;
use core::hash::{BuildHasher, Hash};

use alloc::boxed::Box;
use alloc::vec;

use crate::error::MergeError;
use crate::hash::{DefaultBuildHasher, hash_one, mix64};
use crate::traits::{Insert, Merge, Sketch};

/// A MinHash signature for approximate Jaccard similarity.
///
/// For each of `k` derived hash values, a MinHash sketch keeps the minimum
/// over all inserted items. The probability that two sets produce the same
/// minimum in one position equals their Jaccard similarity, so the fraction
/// of equal positions between two signatures estimates it, with standard
/// error `sqrt(J * (1 - J) / k)`.
///
/// ```text
/// insert("x")
///      |
///      v
///   hash("x") -> d0, d1, ..., d(k-1)   (k derived values)
///
///   minima: [min . min . min . ... . min]
///              ^     ^     ^
///            keep the smallest derived value per position
///
/// jaccard(A, B) ~= |{ i : A.minima[i] == B.minima[i] }| / k
/// ```
///
/// Rather than `k` independent hash functions, the `k` values are derived
/// from one item hash by seeded mixing — the same one-hash-per-operation
/// trade the rest of the crate makes (see
/// [`DoubleHashing`](crate::hash::DoubleHashing) for the classic
/// justification). Sketches merge element-wise with `min`, giving the
/// signature of the union of the two sets.
///
/// # References
///
/// - Andrei Z. Broder, "On the resemblance and containment of documents",
///   Compression and Complexity of Sequences 1997.
///   <https://doi.org/10.1109/SEQUEN.1997.666900>
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MinHash<S = DefaultBuildHasher> {
    minima: Box<[u64]>,
    seed_fingerprint: u64,
    hasher: S,
}

impl MinHash<DefaultBuildHasher> {
    /// Creates a MinHash signature of `num_hashes` minima with seed zero.
    ///
    /// # Panics
    ///
    /// Panics if `num_hashes` is zero.
    pub fn new(num_hashes: usize) -> Self {
        Self::with_seed(num_hashes, 0)
    }

    /// Creates a MinHash signature of `num_hashes` minima with an explicit
    /// hash seed.
    ///
    /// # Panics
    ///
    /// Panics if `num_hashes` is zero.
    pub fn with_seed(num_hashes: usize, seed: u64) -> Self {
        let hasher = DefaultBuildHasher::new(seed);
        Self::from_parts(num_hashes, hasher.seed_fingerprint(), hasher)
    }
}

impl<S> MinHash<S> {
    /// Creates a MinHash sketch from explicit components.
    ///
    /// # Panics
    ///
    /// Panics if `num_hashes` is zero.
    pub fn from_parts(num_hashes: usize, seed_fingerprint: u64, hasher: S) -> Self {
        assert!(num_hashes > 0, "MinHash needs at least one hash");
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

    /// Returns the current signature minima.
    pub fn signature(&self) -> &[u64] {
        &self.minima
    }

    /// Returns the byte length of the signature storage.
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
            self.minima.len(),
            other.minima.len(),
            "MinHash signatures must have the same number of hashes"
        );
        assert_eq!(
            self.seed_fingerprint, other.seed_fingerprint,
            "MinHash signatures must share a hash seed"
        );
    }
}

impl<S> MinHash<S>
where
    S: BuildHasher,
{
    /// Inserts `item`, lowering each minimum whose derived value is smaller.
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

    /// Estimates the Jaccard similarity with `other` as the fraction of
    /// equal signature positions.
    ///
    /// The standard error is `sqrt(J * (1 - J) / k)`, at most `1 / (2 *
    /// sqrt(k))`.
    ///
    /// # Panics
    ///
    /// Panics if the two sketches use different hash counts or seeds.
    pub fn jaccard(&self, other: &Self) -> f64 {
        self.check_compatible(other);
        let equal = self
            .minima
            .iter()
            .zip(other.minima.iter())
            .filter(|(a, b)| a == b)
            .count();
        equal as f64 / self.minima.len() as f64
    }
}

impl<S> Sketch for MinHash<S> {
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

impl<T, S> Insert<T> for MinHash<S>
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

impl<S> Merge for MinHash<S> {
    /// Merges by element-wise minimum, yielding the signature of the union
    /// of the two sets.
    fn merge_from(&mut self, other: &Self) -> Result<(), MergeError> {
        if self.minima.len() != other.minima.len() {
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
    use super::MinHash;
    use crate::error::MergeError;
    use crate::traits::{Insert, Merge};

    #[test]
    fn identical_sets_have_similarity_one() {
        let mut left = MinHash::new(128);
        let mut right = MinHash::new(128);
        for i in 0..1_000_u64 {
            left.insert_item(&i);
            right.insert_item(&i);
        }
        assert_eq!(left.jaccard(&right), 1.0);
    }

    #[test]
    fn disjoint_sets_have_near_zero_similarity() {
        let mut left = MinHash::new(128);
        let mut right = MinHash::new(128);
        for i in 0..1_000_u64 {
            left.insert_item(&i);
            right.insert_item(&(1_000_000 + i));
        }
        assert_eq!(left.jaccard(&right), 0.0);
    }

    #[test]
    fn empty_signatures_match_each_other() {
        let left = MinHash::new(64);
        let right = MinHash::new(64);
        assert_eq!(left.jaccard(&right), 1.0);
    }

    #[test]
    fn merge_is_elementwise_min_and_validates_compatibility() {
        let mut left = MinHash::with_seed(64, 1);
        let mut right = MinHash::with_seed(64, 1);
        left.insert_item(&1_u64);
        right.insert_item(&2_u64);

        let before = left.signature().to_vec();
        left.merge_from(&right).unwrap();
        for (after, (a, b)) in left
            .signature()
            .iter()
            .zip(before.iter().zip(right.signature().iter()))
        {
            assert_eq!(*after, (*a).min(*b));
        }

        let other_k = MinHash::with_seed(32, 1);
        assert_eq!(left.merge_from(&other_k), Err(MergeError::GeometryMismatch));
        let other_seed = MinHash::with_seed(64, 2);
        assert_eq!(left.merge_from(&other_seed), Err(MergeError::SeedMismatch));
    }

    #[test]
    #[should_panic(expected = "must have the same number of hashes")]
    fn jaccard_rejects_different_lengths() {
        let left = MinHash::new(64);
        let right = MinHash::new(128);
        left.jaccard(&right);
    }

    #[test]
    fn capability_traits_work() {
        let mut sketch = MinHash::new(64);
        Insert::<str>::insert(&mut sketch, "alice").unwrap();
        assert!(sketch.signature().iter().any(|&m| m != u64::MAX));
    }
}
