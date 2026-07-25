//! Count Sketch implementation.

use core::convert::Infallible;
use core::hash::{BuildHasher, Hash};

use alloc::vec::Vec;

use crate::block::{Matrix, PackedArray};
use crate::error::MergeError;
#[cfg(any(feature = "std", feature = "libm"))]
use crate::float;
use crate::hash::{DefaultBuildHasher, hash_one, row_index, sign};
use crate::traits::{EstimateCount, Insert, Merge, Sketch};

/// Explicit Count Sketch geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CountSketchGeometry {
    /// Number of signed counters in each row.
    pub width: usize,
    /// Number of independently indexed rows.
    pub depth: usize,
}

/// A Count Sketch for approximate point-frequency queries.
///
/// Count Sketch is similar to Count-Min in that it stores a row matrix and
/// touches one column per row. The difference is that each row also derives a
/// sign, either `+1` or `-1`. Inserts add that sign to the addressed counter.
/// Queries multiply each counter by the same sign and return the median of
/// those signed estimates. This makes collision noise unbiased.
///
/// ```text
/// insert("x")
///      |
///      v
///   hash("x")
///      |
///      +--> row 0 -> col 2, sign +1 -> counter += 1
///      +--> row 1 -> col 5, sign -1 -> counter -= 1
///      +--> row 2 -> col 1, sign +1 -> counter += 1
///
/// estimate("x")
///      |
///      v
///   signed row estimates: [7, 5, 6, 100]
///   median => 6
/// ```
///
/// This implementation stores signed `i64` counters in
/// [`PackedArray<64>`](crate::block::PackedArray) cells using two's-complement
/// representation.
///
/// # References
///
/// - Moses Charikar, Kevin Chen, and Martin Farach-Colton, "Finding Frequent
///   Items in Data Streams", Theoretical Computer Science, 2004.
///   <https://doi.org/10.1016/j.tcs.2003.10.024>
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CountSketch<S = DefaultBuildHasher> {
    rows: Matrix<PackedArray<64>>,
    geometry: CountSketchGeometry,
    seed_fingerprint: u64,
    hasher: S,
    total_count: u64,
}

impl CountSketch<DefaultBuildHasher> {
    /// Creates a Count Sketch with explicit geometry and seed.
    ///
    /// # Panics
    ///
    /// Panics if `geometry` has zero width or zero depth.
    pub fn from_geometry(geometry: CountSketchGeometry, seed: u64) -> Self {
        let hasher = DefaultBuildHasher::new(seed);
        Self::from_parts(geometry, hasher.seed_fingerprint(), hasher)
    }

    /// Creates a Count Sketch from target error parameters.
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

    /// Creates a seeded Count Sketch from target error parameters.
    ///
    /// Available with the `std` or `libm` feature.
    ///
    /// # Panics
    ///
    /// Panics if `epsilon` or `delta` is not finite and in `0.0..1.0`.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn with_error_and_seed(epsilon: f64, delta: f64, seed: u64) -> Self {
        Self::from_geometry(CountSketchGeometry::for_error(epsilon, delta), seed)
    }
}

impl<S> CountSketch<S> {
    /// Creates a Count Sketch from explicit components.
    ///
    /// # Panics
    ///
    /// Panics if `geometry` has zero width or zero depth.
    pub fn from_parts(geometry: CountSketchGeometry, seed_fingerprint: u64, hasher: S) -> Self {
        geometry.validate();
        let rows = (0..geometry.depth)
            .map(|_| PackedArray::<64>::new(geometry.width))
            .collect();
        Self {
            rows: Matrix::from_rows(rows, geometry.width),
            geometry,
            seed_fingerprint,
            hasher,
            total_count: 0,
        }
    }

    /// Returns the realized geometry.
    pub const fn geometry(&self) -> CountSketchGeometry {
        self.geometry
    }

    /// Returns the seed fingerprint used by merge compatibility checks.
    pub const fn seed_fingerprint(&self) -> u64 {
        self.seed_fingerprint
    }

    /// Returns the total number of inserted events, saturated at `u64::MAX`.
    pub const fn total_count(&self) -> u64 {
        self.total_count
    }

    /// Returns the byte length of the counter storage.
    pub fn storage_bytes(&self) -> usize {
        self.rows.iter().map(PackedArray::storage_bytes).sum()
    }

    /// Clears all counters and the total insert count.
    pub fn clear(&mut self) {
        for row in self.rows.iter_mut() {
            row.clear();
        }
        self.total_count = 0;
    }
}

impl<S> CountSketch<S>
where
    S: BuildHasher,
{
    /// Inserts one occurrence of `item`.
    pub fn insert_item<T>(&mut self, item: &T)
    where
        T: Hash + ?Sized,
    {
        self.insert_count(item, 1);
    }

    /// Inserts `item` with weight `count`: each addressed signed counter
    /// changes by `sign * count`. For byte-weighted frequency estimation
    /// instead of looping [`Self::insert_item`].
    pub fn insert_count<T>(&mut self, item: &T, count: u64)
    where
        T: Hash + ?Sized,
    {
        self.insert_signed(item, count.min(i64::MAX as u64) as i64);
    }

    /// Inserts `item` with a signed weight: each addressed signed counter
    /// changes by `sign * count`, so decrements are possible too. The
    /// median-of-rows read-out stays meaningful under signed updates, which
    /// is what distinguishes Count Sketch from Count-Min here.
    pub fn insert_signed<T>(&mut self, item: &T, count: i64)
    where
        T: Hash + ?Sized,
    {
        let hash = hash_one(&self.hasher, item);
        for row in 0..self.geometry.depth {
            let index = row_index(hash, row, self.geometry.width);
            let signed = decode(self.rows.row(row).get(index));
            let delta = sign(hash, row).saturating_mul(count);
            let next = signed.saturating_add(delta);
            self.rows.row_mut(row).set(index, encode(next));
        }
        self.total_count = self.total_count.saturating_add(count.unsigned_abs());
    }

    /// Returns the signed Count Sketch estimate for `item`.
    pub fn estimate_signed<T>(&self, item: &T) -> i64
    where
        T: Hash + ?Sized,
    {
        self.estimate_signed_hash(hash_one(&self.hasher, item))
    }

    /// Returns the signed estimate for the item behind a pre-hashed 64-bit
    /// value, using the same derivation as [`Self::estimate_signed`].
    pub fn estimate_signed_hash(&self, hash: u64) -> i64 {
        let depth = self.geometry.depth;
        // Median of the per-row signed estimates. Typical depths fit a
        // stack buffer, avoiding an allocation per query.
        if depth <= 32 {
            let mut estimates = [0_i64; 32];
            for (row, slot) in estimates.iter_mut().take(depth).enumerate() {
                let index = row_index(hash, row, self.geometry.width);
                *slot = decode(self.rows.row(row).get(index)).saturating_mul(sign(hash, row));
            }
            estimates[..depth].sort_unstable();
            estimates[depth / 2]
        } else {
            let mut estimates: Vec<_> = (0..depth)
                .map(|row| {
                    let index = row_index(hash, row, self.geometry.width);
                    decode(self.rows.row(row).get(index)).saturating_mul(sign(hash, row))
                })
                .collect();
            estimates.sort_unstable();
            estimates[depth / 2]
        }
    }

    /// Returns the non-negative frequency estimate for `item`.
    pub fn estimate_item<T>(&self, item: &T) -> u64
    where
        T: Hash + ?Sized,
    {
        self.estimate_signed(item).max(0) as u64
    }
}

impl CountSketchGeometry {
    /// Computes Count Sketch geometry from target error parameters.
    ///
    /// Uses `width = ceil(3 / epsilon^2)` and
    /// `depth = ceil(ln(1 / delta))`.
    ///
    /// Available with the `std` or `libm` feature.
    ///
    /// # Panics
    ///
    /// Panics if `epsilon` or `delta` is not finite and in `0.0..1.0`.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn for_error(epsilon: f64, delta: f64) -> Self {
        assert!(
            epsilon.is_finite() && epsilon > 0.0 && epsilon < 1.0,
            "epsilon must be finite and in 0.0..1.0"
        );
        assert!(
            delta.is_finite() && delta > 0.0 && delta < 1.0,
            "delta must be finite and in 0.0..1.0"
        );
        let width = float::ceil(3.0 / (epsilon * epsilon));
        let depth = float::ceil(float::ln(1.0 / delta));
        assert!(
            width <= usize::MAX as f64 && depth <= usize::MAX as f64,
            "computed Count Sketch geometry does not fit in usize"
        );
        Self {
            width: width.max(1.0) as usize,
            depth: depth.max(1.0) as usize,
        }
    }

    /// Validates that both geometry dimensions are non-zero.
    ///
    /// # Panics
    ///
    /// Panics if `width` or `depth` is zero.
    pub fn validate(self) {
        assert!(
            self.width > 0,
            "Count Sketch width must be greater than zero"
        );
        assert!(
            self.depth > 0,
            "Count Sketch depth must be greater than zero"
        );
    }
}

impl<T, S> crate::traits::Estimator<T> for CountSketch<S>
where
    T: Hash + ?Sized,
    S: BuildHasher,
{
    fn estimate(&self, item: &T) -> u64 {
        self.estimate_item(item)
    }

    fn insert_count(&mut self, item: &T, count: u64) {
        Self::insert_count(self, item, count);
    }

    fn total(&self) -> u64 {
        self.total_count()
    }
}

impl<S> Sketch for CountSketch<S> {
    fn clear(&mut self) {
        self.clear();
    }

    fn len_hint(&self) -> Option<u64> {
        Some(self.total_count)
    }

    fn storage_bytes(&self) -> usize {
        self.storage_bytes()
    }
}

impl<T, S> Insert<T> for CountSketch<S>
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

impl<T, S> EstimateCount<T> for CountSketch<S>
where
    T: Hash + ?Sized,
    S: BuildHasher,
{
    fn estimate(&self, item: &T) -> u64 {
        self.estimate_item(item)
    }
}

impl<S> Merge for CountSketch<S> {
    fn merge_from(&mut self, other: &Self) -> Result<(), MergeError> {
        if self.geometry != other.geometry {
            return Err(MergeError::GeometryMismatch);
        }
        if self.seed_fingerprint != other.seed_fingerprint {
            return Err(MergeError::SeedMismatch);
        }
        for row in 0..self.geometry.depth {
            self.rows
                .row_mut(row)
                .merge_with(other.rows.row(row), |a, b| {
                    encode(decode(a).saturating_add(decode(b)))
                });
        }
        self.total_count = self.total_count.saturating_add(other.total_count);
        Ok(())
    }
}

fn encode(value: i64) -> u64 {
    value as u64
}

fn decode(value: u64) -> i64 {
    value as i64
}

#[cfg(test)]
mod tests {
    use super::{CountSketch, CountSketchGeometry};
    use crate::traits::{EstimateCount, Insert, Merge};

    #[test]
    fn estimates_inserted_counts() {
        let mut sketch = CountSketch::with_error(0.25, 0.01);
        for _ in 0..8 {
            sketch.insert_item("a");
        }
        for _ in 0..3 {
            sketch.insert_item("b");
        }

        assert!(sketch.estimate_item("a") > 0);
        assert!(EstimateCount::<str>::estimate(&sketch, "b") > 0);
        assert_eq!(sketch.total_count(), 11);
    }

    #[test]
    fn insert_trait_works() {
        let mut sketch = CountSketch::with_error(0.25, 0.01);
        Insert::<str>::insert(&mut sketch, "a").unwrap();
        assert!(sketch.estimate_item("a") > 0);
    }

    #[test]
    fn weighted_and_signed_inserts_behave() {
        let mut weighted = CountSketch::with_error(0.1, 0.01);
        let mut iterated = CountSketch::with_error(0.1, 0.01);
        weighted.insert_count("a", 9);
        for _ in 0..9 {
            iterated.insert_item("a");
        }
        assert_eq!(weighted.estimate_signed("a"), iterated.estimate_signed("a"));
        assert_eq!(weighted.total_count(), 9);

        // Signed inserts can decrement; gross volume still accumulates.
        weighted.insert_signed("a", -4);
        assert!(weighted.estimate_signed("a") <= 9);
        assert_eq!(weighted.total_count(), 13);
    }

    #[test]
    fn signed_insert_handles_i64_min_without_panicking() {
        let mut sketch = CountSketch::with_error(0.1, 0.01);
        sketch.insert_item("a");
        // count == i64::MIN would overflow a naive sign * count multiply in
        // debug builds; saturating_mul keeps it defined.
        sketch.insert_signed("a", i64::MIN);
        assert_eq!(sketch.total_count(), 1 + i64::MIN.unsigned_abs());
    }

    #[test]
    fn merge_combines_signed_rows() {
        let geometry = CountSketchGeometry {
            width: 256,
            depth: 5,
        };
        let mut left = CountSketch::from_geometry(geometry, 11);
        let mut right = CountSketch::from_geometry(geometry, 11);
        left.insert_item("a");
        right.insert_item("a");

        left.merge_from(&right).unwrap();
        assert!(left.estimate_item("a") >= 2);
    }
}
