//! Theta sketch (KMV / bottom-k) for cardinality and set operations.

use core::convert::Infallible;
use core::hash::{BuildHasher, Hash};

use alloc::vec::Vec;

use crate::error::MergeError;
use crate::hash::{DefaultBuildHasher, hash_one};
use crate::traits::{EstimateCardinality, Insert, Merge, Sketch};

/// A theta sketch for distinct counts with set algebra: union,
/// intersection, difference, and Jaccard similarity.
///
/// The sketch keeps the `k` smallest item hashes seen so far, deduplicated
/// and sorted. Since hashes are uniform in `[0, 2^64)`, the `k`-th smallest
/// is a threshold `theta`: about a `theta / 2^64` fraction of all possible
/// hashes falls below it, so the distinct count is estimated as
/// `(k - 1) / theta`. While fewer than `k` values are retained the sketch
/// is *exact*. The standard error is about `1 / sqrt(k)`.
///
/// ```text
/// insert("x") -> hash("x") -> insert into sorted retained, keep k smallest
///
///   retained: [h1, h2, ..., hk)         theta = hk / 2^64
///
/// union:        k smallest of both retained lists
/// intersection: values in both lists, estimated count / min(thetaA, thetaB)
/// difference:   values in A but not B, same threshold
/// ```
///
/// The same bottom-k principle makes set operations work where HyperLogLog
/// cannot: two sketches can be intersected or subtracted, not just merged.
///
/// # References
///
/// - Ziv Bar-Yossef, T. S. Jayram, Ravi Kumar, D. Sivakumar, and Luca
///   Trevisan, "Counting Distinct Elements in a Data Stream", RANDOM 2002.
///   <https://doi.org/10.1007/3-540-45726-7_1>
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ThetaSketch<S = DefaultBuildHasher> {
    retained: Vec<u64>,
    k: usize,
    seed_fingerprint: u64,
    hasher: S,
}

/// `2^64` as `f64`: the size of the hash space.
const HASH_SPACE: f64 = 18_446_744_073_709_551_616.0;

impl ThetaSketch<DefaultBuildHasher> {
    /// Creates a theta sketch retaining `k` hashes, with seed zero.
    ///
    /// # Panics
    ///
    /// Panics if `k` is zero.
    pub fn new(k: usize) -> Self {
        Self::with_seed(k, 0)
    }

    /// Creates a theta sketch retaining `k` hashes, with an explicit seed.
    ///
    /// # Panics
    ///
    /// Panics if `k` is zero.
    pub fn with_seed(k: usize, seed: u64) -> Self {
        let hasher = DefaultBuildHasher::new(seed);
        Self::from_parts(k, hasher.seed_fingerprint(), hasher)
    }
}

impl<S> ThetaSketch<S> {
    /// Creates a theta sketch from explicit components.
    ///
    /// # Panics
    ///
    /// Panics if `k` is zero.
    pub fn from_parts(k: usize, seed_fingerprint: u64, hasher: S) -> Self {
        assert!(k > 0, "theta sketch needs at least one retained hash");
        Self {
            retained: Vec::new(),
            k,
            seed_fingerprint,
            hasher,
        }
    }

    /// Returns the number of retained hashes the sketch holds at most.
    pub const fn k(&self) -> usize {
        self.k
    }

    /// Returns the seed fingerprint used by compatibility checks.
    pub const fn seed_fingerprint(&self) -> u64 {
        self.seed_fingerprint
    }

    /// Returns the retained hashes, sorted ascending.
    pub fn retained(&self) -> &[u64] {
        &self.retained
    }

    /// Returns whether the sketch is exact: fewer than `k` distinct hashes
    /// have been seen, so the retained list is the complete set.
    pub fn is_exact(&self) -> bool {
        self.retained.len() < self.k
    }

    /// Returns the sampling threshold as a fraction of the hash space in
    /// `0.0..=1.0`; `1.0` while the sketch is exact.
    pub fn theta(&self) -> f64 {
        if self.is_exact() {
            1.0
        } else {
            self.retained[self.k - 1] as f64 / HASH_SPACE
        }
    }

    /// Returns the theoretical standard error of the cardinality estimate,
    /// about `1 / sqrt(k - 1)`.
    ///
    /// Available with the `std` or `libm` feature.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn standard_error(&self) -> f64 {
        1.0 / crate::float::sqrt((self.k.saturating_sub(1) as f64).max(1.0))
    }

    /// Returns the byte length of the retained-hash storage.
    pub fn storage_bytes(&self) -> usize {
        self.retained.len() * size_of::<u64>()
    }

    /// Clears the sketch.
    pub fn clear(&mut self) {
        self.retained.clear();
    }

    /// Checks that two sketches can be combined.
    ///
    /// # Panics
    ///
    /// Panics if the sketches use different `k` or different hash seeds.
    fn check_compatible(&self, other: &Self) {
        assert_eq!(self.k, other.k, "theta sketches must retain the same k");
        assert_eq!(
            self.seed_fingerprint, other.seed_fingerprint,
            "theta sketches must share a hash seed"
        );
    }

    /// Inserts a precomputed hash, maintaining the sorted `k` smallest.
    fn insert_hash(&mut self, hash: u64) {
        match self.retained.binary_search(&hash) {
            Ok(_) => {}
            Err(position) => {
                self.retained.insert(position, hash);
                self.retained.truncate(self.k);
            }
        }
    }

    /// Counts retained values that also appear in `other`'s retained list
    /// (both lists are sorted, so this is a linear merge-join).
    fn count_common(&self, other: &Self) -> usize {
        let mut common = 0_usize;
        let mut left = self.retained.iter().peekable();
        let mut right = other.retained.iter().peekable();
        while let (Some(&&a), Some(&&b)) = (left.peek(), right.peek()) {
            match a.cmp(&b) {
                core::cmp::Ordering::Less => {
                    left.next();
                }
                core::cmp::Ordering::Greater => {
                    right.next();
                }
                core::cmp::Ordering::Equal => {
                    common += 1;
                    left.next();
                    right.next();
                }
            }
        }
        common
    }
}

impl<S> ThetaSketch<S>
where
    S: BuildHasher,
{
    /// Inserts `item`.
    pub fn insert_item<T>(&mut self, item: &T)
    where
        T: Hash + ?Sized,
    {
        let hash = hash_one(&self.hasher, item);
        self.insert_hash(hash);
    }

    /// Estimates the number of distinct inserted items.
    pub fn cardinality(&self) -> f64 {
        if self.is_exact() {
            self.retained.len() as f64
        } else {
            (self.k - 1) as f64 / self.theta()
        }
    }

    /// Estimates `|A ∪ B|`: the cardinality of the merged bottom-k sample.
    ///
    /// # Panics
    ///
    /// Panics if the sketches use different `k` or different hash seeds.
    pub fn estimate_union(&self, other: &Self) -> f64 {
        self.check_compatible(other);
        let mut merged = self.retained.clone();
        for &hash in &other.retained {
            if let Err(position) = merged.binary_search(&hash) {
                merged.insert(position, hash);
            }
        }
        merged.truncate(self.k);
        if merged.len() < self.k {
            merged.len() as f64
        } else {
            (self.k - 1) as f64 / (merged[self.k - 1] as f64 / HASH_SPACE)
        }
    }

    /// Estimates `|A ∩ B|`: the common retained values below
    /// `min(thetaA, thetaB)`, scaled by that threshold.
    ///
    /// # Panics
    ///
    /// Panics if the sketches use different `k` or different hash seeds.
    pub fn estimate_intersection(&self, other: &Self) -> f64 {
        self.check_compatible(other);
        let theta = self.theta().min(other.theta());
        self.count_common(other) as f64 / theta
    }

    /// Estimates `|A \ B|`: the retained values of `self` absent from
    /// `other`, scaled by `min(thetaA, thetaB)`.
    ///
    /// # Panics
    ///
    /// Panics if the sketches use different `k` or different hash seeds.
    pub fn estimate_difference(&self, other: &Self) -> f64 {
        self.check_compatible(other);
        let theta = self.theta().min(other.theta());
        let common = self.count_common(other);
        (self.retained.len() - common) as f64 / theta
    }

    /// Estimates the Jaccard similarity `|A ∩ B| / |A ∪ B|` as the ratio of
    /// common to merged retained values.
    ///
    /// # Panics
    ///
    /// Panics if the sketches use different `k` or different hash seeds.
    pub fn jaccard(&self, other: &Self) -> f64 {
        self.check_compatible(other);
        let common = self.count_common(other) as f64;
        if common == 0.0 {
            return 0.0;
        }
        let merged_len = {
            let mut merged = self.retained.clone();
            for &hash in &other.retained {
                if let Err(position) = merged.binary_search(&hash) {
                    merged.insert(position, hash);
                }
            }
            merged.len()
        };
        common / merged_len as f64
    }
}

impl<S> Sketch for ThetaSketch<S> {
    fn clear(&mut self) {
        self.clear();
    }

    fn len_hint(&self) -> Option<u64> {
        if self.is_exact() {
            Some(self.retained.len() as u64)
        } else {
            None
        }
    }

    fn storage_bytes(&self) -> usize {
        self.storage_bytes()
    }
}

impl<T, S> Insert<T> for ThetaSketch<S>
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

impl<S> EstimateCardinality for ThetaSketch<S>
where
    S: BuildHasher,
{
    fn cardinality(&self) -> f64 {
        self.cardinality()
    }
}

impl<S> Merge for ThetaSketch<S>
where
    S: BuildHasher,
{
    /// Merges by keeping the `k` smallest hashes of both retained lists —
    /// the union's bottom-k sample. Requires equal `k` and seed.
    fn merge_from(&mut self, other: &Self) -> Result<(), MergeError> {
        if self.k != other.k {
            return Err(MergeError::GeometryMismatch);
        }
        if self.seed_fingerprint != other.seed_fingerprint {
            return Err(MergeError::SeedMismatch);
        }
        for &hash in &other.retained {
            self.insert_hash(hash);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::ThetaSketch;
    use crate::error::MergeError;
    use crate::traits::{Insert, Merge, Sketch};

    #[test]
    fn exact_below_capacity() {
        let mut sketch = ThetaSketch::new(16);
        for i in 0..10_u64 {
            sketch.insert_item(&i);
            sketch.insert_item(&i); // duplicates are deduplicated
        }
        assert!(sketch.is_exact());
        assert_eq!(sketch.cardinality(), 10.0);
        assert_eq!(Sketch::len_hint(&sketch), Some(10));
    }

    #[test]
    fn retained_stays_sorted_and_capped() {
        let mut sketch = ThetaSketch::new(8);
        for i in 0..1_000_u64 {
            sketch.insert_item(&i);
        }
        let retained = sketch.retained();
        assert_eq!(retained.len(), 8);
        assert!(retained.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn merge_keeps_k_smallest_and_validates() {
        let mut left = ThetaSketch::with_seed(16, 1);
        let mut right = ThetaSketch::with_seed(16, 1);
        for i in 0..1_000_u64 {
            left.insert_item(&i);
            right.insert_item(&(1_000 + i));
        }
        left.merge_from(&right).unwrap();
        assert_eq!(left.retained().len(), 16);

        let other_k = ThetaSketch::with_seed(8, 1);
        assert_eq!(left.merge_from(&other_k), Err(MergeError::GeometryMismatch));
        let other_seed = ThetaSketch::with_seed(16, 2);
        assert_eq!(left.merge_from(&other_seed), Err(MergeError::SeedMismatch));
    }

    #[test]
    fn exact_set_operations() {
        // Both sketches exact: set operations are exact too.
        let mut a = ThetaSketch::new(100);
        let mut b = ThetaSketch::new(100);
        for i in 0..50_u64 {
            a.insert_item(&i);
        }
        for i in 25..75_u64 {
            b.insert_item(&i);
        }
        assert_eq!(a.estimate_intersection(&b), 25.0);
        assert_eq!(a.estimate_union(&b), 75.0);
        assert_eq!(a.estimate_difference(&b), 25.0);
        assert!((a.jaccard(&b) - 25.0 / 75.0).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "must retain the same k")]
    fn set_operations_validate_k() {
        let a = ThetaSketch::new(16);
        let b = ThetaSketch::new(32);
        a.estimate_intersection(&b);
    }

    #[test]
    fn capability_traits_work() {
        let mut sketch = ThetaSketch::new(16);
        Insert::<str>::insert(&mut sketch, "alice").unwrap();
        assert!(sketch.cardinality() >= 1.0);
    }
}
