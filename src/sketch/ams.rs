//! AMS tug-of-war sketch for the second frequency moment.

use core::convert::Infallible;
use core::hash::{BuildHasher, Hash};

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use crate::error::MergeError;
#[cfg(any(feature = "std", feature = "libm"))]
use crate::float;
use crate::hash::{DefaultBuildHasher, hash_one, row_index, sign};
use crate::traits::{Insert, Merge, Sketch};

/// An AMS-family sketch for the second frequency moment `F2`: the
/// self-join size `sum(f_i^2)`, i.e. the squared L2 norm of the frequency
/// vector.
///
/// Every item maps to one bucket in each of `groups` rows and adds a
/// per-row sign there; the sum of squared counters in a row is an
/// unbiased estimator of `F2` (the `+/-1` cross terms vanish in
/// expectation). The estimate is the median over groups of those sums —
/// the median-of-means amplification from the AMS paper, in the
/// bucketed-per-group layout of Charikar's Count Sketch, so an insert
/// costs `O(groups)` writes instead of `O(groups × width)`.
///
/// ```text
/// insert("x"): for each group g: counters[g][h_g("x")] += sign_g("x")
///
/// f2() = median over groups g of sum_j(counters[g][j]^2)
/// l2_norm() = sqrt(f2())
/// ```
///
/// The sketch is *linear*: merging is an element-wise sum of the row
/// counters and yields exactly the sketch of the combined stream.
///
/// # Rényi entropy
///
/// `F2` directly yields the order-2 Rényi entropy of the stream's
/// distribution: `H2 = -log2(F2 / N^2)`. It is exact for the uniform
/// distribution (`log2(n)` for `n` distinct single items) and tracks
/// Shannon entropy closely for detection-style metrics (`H2 <= H_shannon`
/// always). Multiplicative Shannon-entropy estimation in small space is a
/// research problem of its own and deliberately out of scope here; see the
/// composition note on [`Self::renyi2_entropy`].
///
/// # References
///
/// - Noga Alon, Yossi Matias, and Mario Szegedy, "The Space Complexity of
///   Approximating the Frequency Moments", STOC 1996 (journal version:
///   JCSS 58(1), 1999). <https://doi.org/10.1145/237814.237823>
/// - Moses Charikar, Kevin Chen, and Martin Farach-Colton, "Finding
///   Frequent Items in Data Streams", Theoretical Computer Science, 2004.
///   <https://doi.org/10.1016/j.tcs.2003.10.024>
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AmsSketch<S = DefaultBuildHasher> {
    counters: Box<[i64]>,
    groups: usize,
    width: usize,
    total: u64,
    seed_fingerprint: u64,
    hasher: S,
}

impl AmsSketch<DefaultBuildHasher> {
    /// Creates an AMS sketch with explicit dimensions and seed zero.
    ///
    /// # Panics
    ///
    /// Panics if `groups` or `width` is zero.
    pub fn new(groups: usize, width: usize) -> Self {
        Self::with_seed(groups, width, 0)
    }

    /// Creates an AMS sketch with explicit dimensions and hash seed.
    ///
    /// # Panics
    ///
    /// Panics if `groups` or `width` is zero.
    pub fn with_seed(groups: usize, width: usize, seed: u64) -> Self {
        let hasher = DefaultBuildHasher::new(seed);
        Self::from_parts(groups, width, hasher.seed_fingerprint(), hasher)
    }

    /// Creates an AMS sketch solved from target error parameters.
    ///
    /// Uses `width = ceil(16 / epsilon^2)` counters per group and
    /// `groups = ceil(2 * ln(1 / delta))`, the paper's constants: the
    /// relative error stays below `epsilon` with probability at least
    /// `1 - delta`.
    ///
    /// Available with the `std` or `libm` feature.
    ///
    /// # Panics
    ///
    /// Panics if `epsilon` or `delta` is not finite and in `0.0..1.0`.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn with_error(epsilon: f64, delta: f64) -> Self {
        Self::with_error_and_seed(epsilon, delta, 0)
    }

    /// Creates a seeded AMS sketch solved from target error parameters.
    ///
    /// Available with the `std` or `libm` feature.
    ///
    /// # Panics
    ///
    /// Panics if `epsilon` or `delta` is not finite and in `0.0..1.0`.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn with_error_and_seed(epsilon: f64, delta: f64, seed: u64) -> Self {
        assert!(
            epsilon.is_finite() && epsilon > 0.0 && epsilon < 1.0,
            "epsilon must be finite and in 0.0..1.0"
        );
        assert!(
            delta.is_finite() && delta > 0.0 && delta < 1.0,
            "delta must be finite and in 0.0..1.0"
        );
        let width = float::ceil(16.0 / (epsilon * epsilon)) as usize;
        let groups = float::ceil(2.0 * float::ln(1.0 / delta)) as usize;
        Self::with_seed(groups.max(1), width.max(1), seed)
    }
}

impl<S> AmsSketch<S> {
    /// Creates an AMS sketch from explicit components.
    ///
    /// # Panics
    ///
    /// Panics if `groups` or `width` is zero.
    pub fn from_parts(groups: usize, width: usize, seed_fingerprint: u64, hasher: S) -> Self {
        assert!(groups > 0, "AMS group count must be greater than zero");
        assert!(width > 0, "AMS width must be greater than zero");
        Self {
            counters: vec![0; groups * width].into_boxed_slice(),
            groups,
            width,
            total: 0,
            seed_fingerprint,
            hasher,
        }
    }

    /// Returns the number of counter groups.
    pub const fn groups(&self) -> usize {
        self.groups
    }

    /// Returns the number of counters per group.
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Returns the total number of inserted events.
    pub const fn total_count(&self) -> u64 {
        self.total
    }

    /// Returns the seed fingerprint used by merge compatibility checks.
    pub const fn seed_fingerprint(&self) -> u64 {
        self.seed_fingerprint
    }

    /// Returns the current tug-of-war sums, laid out group-major.
    pub fn counters(&self) -> &[i64] {
        &self.counters
    }

    /// Returns the byte length of the counter storage.
    pub fn storage_bytes(&self) -> usize {
        self.counters.len() * size_of::<i64>()
    }

    /// Clears all counters and the total count.
    pub fn clear(&mut self) {
        self.counters.fill(0);
        self.total = 0;
    }

    /// Returns the group sums of squared counters, used by [`Self::f2`].
    /// For the bucketed layout the estimator per group is the *sum* (not
    /// the mean) of squared counters: `E[sum(z_j^2)] = F2` per row.
    fn group_sums(&self) -> Vec<f64> {
        self.counters
            .chunks_exact(self.width)
            .map(|group| {
                group
                    .iter()
                    .map(|&z| (z as f64) * (z as f64))
                    .sum::<f64>()
            })
            .collect()
    }
}

impl<S> AmsSketch<S>
where
    S: BuildHasher,
{
    /// Inserts one occurrence of `item`, adding its sign into one bucket
    /// per group — `O(groups)` writes, not `O(groups × width)`.
    ///
    /// The per-group bucketing is the Charikar–Chen–Farach-Colton layout:
    /// an item maps to a single counter in each row and adds its sign
    /// there. The sum of squared counters within a row is unbiased for
    /// `F2` by the same `+/-1` orthogonality as the classic tug-of-war
    /// layout, with the same median-of-groups amplification — identical
    /// geometry and error guarantees, at a fraction of the insert cost.
    pub fn insert_item<T>(&mut self, item: &T)
    where
        T: Hash + ?Sized,
    {
        let hash = hash_one(&self.hasher, item);
        for group in 0..self.groups {
            let index = group * self.width + row_index(hash, group, self.width);
            self.counters[index] += sign(hash, group);
        }
        self.total = self.total.saturating_add(1);
    }

    /// Estimates `F2 = sum(f_i^2)`, the self-join size of the stream.
    pub fn f2(&self) -> f64 {
        let mut sums = self.group_sums();
        sums.sort_by(f64::total_cmp);
        sums[sums.len() / 2]
    }

    /// Estimates the L2 norm `sqrt(sum(f_i^2))` of the frequency vector.
    ///
    /// Available with the `std` or `libm` feature.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn l2_norm(&self) -> f64 {
        float::sqrt(self.f2())
    }

    /// Estimates the order-2 Rényi entropy of the stream's frequency
    /// distribution, `H2 = -log2(F2 / N^2)` bits.
    ///
    /// Exact for the uniform distribution (`log2(n)` for `n` distinct
    /// single items). For Shannon entropy specifically: `H2 <= H_shannon`,
    /// and the two track closely on detection-style workloads — a common
    /// composition is to report `H2` plus heavy-hitter estimates from
    /// [`crate::sketch::TopK`] when a Shannon reading is needed. First-class
    /// multiplicative Shannon estimation in small space (Guha–McGregor–
    /// Venkat-style sketches) is deliberately out of scope. Returns 0.0 for
    /// an empty sketch.
    ///
    /// Available with the `std` or `libm` feature.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn renyi2_entropy(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let ln2 = core::f64::consts::LN_2;
        (2.0 * float::ln(self.total as f64) - float::ln(self.f2())) / ln2
    }
}

impl<S> Sketch for AmsSketch<S> {
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

impl<T, S> Insert<T> for AmsSketch<S>
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

impl<S> Merge for AmsSketch<S> {
    /// Merges by element-wise sum of the counters: the result is exactly
    /// the sketch of the combined stream.
    fn merge_from(&mut self, other: &Self) -> Result<(), MergeError> {
        if self.groups != other.groups || self.width != other.width {
            return Err(MergeError::GeometryMismatch);
        }
        if self.seed_fingerprint != other.seed_fingerprint {
            return Err(MergeError::SeedMismatch);
        }
        for (left, right) in self.counters.iter_mut().zip(other.counters.iter()) {
            *left += right;
        }
        self.total = self.total.saturating_add(other.total);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::AmsSketch;
    use crate::error::MergeError;
    use crate::traits::{Insert, Merge};

    #[test]
    fn empty_sketch_estimates_zero() {
        let sketch = AmsSketch::new(4, 64);
        assert_eq!(sketch.f2(), 0.0);
        assert_eq!(sketch.l2_norm(), 0.0);
    }

    #[test]
    fn single_item_f2_is_exact() {
        // One distinct item inserted n times: z_j = +/-n per row, so every
        // squared counter is n^2 and the estimate is exact.
        let mut sketch = AmsSketch::new(4, 64);
        for _ in 0..50 {
            sketch.insert_item(&"same");
        }
        assert_eq!(sketch.f2(), 2_500.0);
        assert_eq!(sketch.l2_norm(), 50.0);
    }

    #[test]
    fn renyi2_matches_uniform_within_estimator_noise() {
        // n distinct items once each: true F2 = n, so true H2 = log2(n) —
        // the Rényi-2 ceiling. The AMS estimate is unbiased but noisy, so
        // the check is a bound, not equality.
        let mut sketch = AmsSketch::new(4, 64);
        for i in 0..1_000_u64 {
            sketch.insert_item(&i);
        }
        assert_eq!(sketch.total_count(), 1_000);
        let entropy = sketch.renyi2_entropy();
        let expected = (1_000_f64).log2();
        assert!(
            (entropy - expected).abs() <= 1.0,
            "H2 {entropy} vs true {expected}"
        );
    }

    #[test]
    fn renyi2_is_zero_for_empty_and_degenerate_streams() {
        let mut sketch = AmsSketch::new(4, 64);
        assert_eq!(sketch.renyi2_entropy(), 0.0);
        for _ in 0..100 {
            sketch.insert_item(&"same");
        }
        // One outcome: F2 = N^2, H2 = 0.
        assert_eq!(sketch.renyi2_entropy(), 0.0);
        sketch.clear();
        assert_eq!(sketch.total_count(), 0);
        assert_eq!(sketch.renyi2_entropy(), 0.0);
    }

    #[test]
    fn merge_adds_totals() {
        let mut left = AmsSketch::with_seed(3, 32, 5);
        let mut right = AmsSketch::with_seed(3, 32, 5);
        left.insert_item(&1_u64);
        right.insert_item(&2_u64);
        right.insert_item(&2_u64);
        left.merge_from(&right).unwrap();
        assert_eq!(left.total_count(), 3);
    }

    #[test]
    fn merge_adds_counters_and_validates() {
        let mut left = AmsSketch::with_seed(3, 32, 5);
        let mut right = AmsSketch::with_seed(3, 32, 5);
        left.insert_item(&1_u64);
        right.insert_item(&2_u64);

        let before: Vec<i64> = left.counters().to_vec();
        left.merge_from(&right).unwrap();
        for (after, (a, b)) in left
            .counters()
            .iter()
            .zip(before.iter().zip(right.counters().iter()))
        {
            assert_eq!(*after, *a + *b);
        }

        let other_geometry = AmsSketch::with_seed(4, 32, 5);
        assert_eq!(
            left.merge_from(&other_geometry),
            Err(MergeError::GeometryMismatch)
        );
        let other_seed = AmsSketch::with_seed(3, 32, 6);
        assert_eq!(left.merge_from(&other_seed), Err(MergeError::SeedMismatch));
    }

    #[test]
    fn capability_traits_work() {
        let mut sketch = AmsSketch::new(2, 16);
        Insert::<str>::insert(&mut sketch, "alice").unwrap();
        assert!(sketch.f2() >= 1.0);
    }

    #[test]
    fn with_error_solves_dimensions() {
        let sketch = AmsSketch::with_error(0.1, 0.05);
        assert_eq!(sketch.width(), 1_600);
        assert!(sketch.groups() >= 6);
    }
}
