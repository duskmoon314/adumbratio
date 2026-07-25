//! Count-Min Sketch implementation.

use core::convert::Infallible;
use core::hash::{BuildHasher, Hash};
use core::marker::PhantomData;


use crate::block::{Matrix, PackedArray};
use crate::error::MergeError;
#[cfg(any(feature = "std", feature = "libm"))]
use crate::float;
use crate::hash::{DefaultBuildHasher, hash_one, row_index};
use crate::policy::{ConservativeUpdate, PlainUpdate};
use crate::traits::{EstimateCount, Insert, Merge, Sketch};

/// Explicit Count-Min geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CountMinGeometry {
    /// Number of counters in each row.
    pub width: usize,
    /// Number of independently indexed rows.
    pub depth: usize,
}

/// A Count-Min Sketch for approximate point-frequency queries.
///
/// Count-Min stores a matrix of unsigned counters. Each inserted item touches
/// one counter per row. A query reads the same counters and returns their
/// minimum. Counter collisions can only increase cells, so the estimate never
/// underestimates when counters do not saturate.
///
/// ```text
/// insert("x")
///      |
///      v
///   hash("x")
///      |
///      +--> row 0 -> col 4 -> increment
///      +--> row 1 -> col 1 -> increment
///      +--> row 2 -> col 7 -> increment
///
/// rows:
///   r0 [0 0 0 0 5 0 0 0]
///   r1 [0 3 0 0 0 0 0 0]
///   r2 [0 0 0 0 0 0 0 4]
///
/// estimate("x") = min(5, 3, 4) = 3
/// ```
///
/// With [`PlainUpdate`], every addressed cell is incremented. With
/// [`ConservativeUpdate`], the sketch first reads all addressed cells and
/// increments only cells equal to the current minimum, reducing overestimation
/// bias while preserving the same query path.
///
/// # Saturation
///
/// Counters pin at `2^BITS - 1` instead of wrapping. With the default
/// `BITS = 32` this is unreachable in practice; with narrow counters a
/// pinned cell stops counting and estimates may underestimate. Monitor
/// [`CountMinSketch::saturated_cells`] when using narrow counters.
///
/// # References
///
/// - Graham Cormode and S. Muthukrishnan, "An Improved Data Stream Summary:
///   The Count-Min Sketch and its Applications", Journal of Algorithms, 2005.
///   <https://doi.org/10.1016/j.jalgor.2003.12.001>
/// - Cristian Estan and George Varghese, "New Directions in Traffic
///   Measurement and Accounting", SIGCOMM 2002. <https://doi.org/10.1145/633025.633056>
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CountMinSketch<const BITS: u32 = 32, S = DefaultBuildHasher, U = PlainUpdate> {
    rows: Matrix<PackedArray<BITS>>,
    geometry: CountMinGeometry,
    seed_fingerprint: u64,
    hasher: S,
    update: PhantomData<U>,
    total_count: u64,
}

impl CountMinSketch<32, DefaultBuildHasher, PlainUpdate> {
    /// Creates a plain Count-Min Sketch with explicit geometry and seed.
    ///
    /// # Panics
    ///
    /// Panics if `geometry` has zero width or zero depth.
    pub fn from_geometry(geometry: CountMinGeometry, seed: u64) -> Self {
        let hasher = DefaultBuildHasher::new(seed);
        Self::from_parts(geometry, hasher.seed_fingerprint(), hasher, PlainUpdate)
    }

    /// Creates a plain Count-Min Sketch from target error parameters.
    ///
    /// `epsilon` controls additive error and `delta` controls failure
    /// probability.
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

    /// Creates a seeded plain Count-Min Sketch from target error parameters.
    ///
    /// Available with the `std` or `libm` feature.
    ///
    /// # Panics
    ///
    /// Panics if `epsilon` or `delta` is not finite and in `0.0..1.0`.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn with_error_and_seed(epsilon: f64, delta: f64, seed: u64) -> Self {
        Self::from_geometry(CountMinGeometry::for_error(epsilon, delta), seed)
    }

    /// Creates a conservative-update Count-Min Sketch with explicit geometry.
    ///
    /// # Panics
    ///
    /// Panics if `geometry` has zero width or zero depth.
    pub fn conservative_from_geometry(
        geometry: CountMinGeometry,
        seed: u64,
    ) -> CountMinSketch<32, DefaultBuildHasher, ConservativeUpdate> {
        let hasher = DefaultBuildHasher::new(seed);
        CountMinSketch::from_parts(
            geometry,
            hasher.seed_fingerprint(),
            hasher,
            ConservativeUpdate,
        )
    }

    /// Creates a conservative-update Count-Min Sketch from target parameters.
    ///
    /// Available with the `std` or `libm` feature.
    ///
    /// # Panics
    ///
    /// Panics if `epsilon` or `delta` is not finite and in `0.0..1.0`.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn conservative_with_error(
        epsilon: f64,
        delta: f64,
    ) -> CountMinSketch<32, DefaultBuildHasher, ConservativeUpdate> {
        Self::conservative_with_error_and_seed(epsilon, delta, 0)
    }

    /// Creates a seeded conservative-update Count-Min Sketch.
    ///
    /// Available with the `std` or `libm` feature.
    ///
    /// # Panics
    ///
    /// Panics if `epsilon` or `delta` is not finite and in `0.0..1.0`.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn conservative_with_error_and_seed(
        epsilon: f64,
        delta: f64,
        seed: u64,
    ) -> CountMinSketch<32, DefaultBuildHasher, ConservativeUpdate> {
        Self::conservative_from_geometry(CountMinGeometry::for_error(epsilon, delta), seed)
    }
}

impl<const BITS: u32, S, U> CountMinSketch<BITS, S, U> {
    /// Creates a Count-Min Sketch from explicit components.
    ///
    /// # Panics
    ///
    /// Panics if `geometry` has zero width or zero depth, or if `BITS` is not
    /// in `1..=64`.
    pub fn from_parts(
        geometry: CountMinGeometry,
        seed_fingerprint: u64,
        hasher: S,
        update: U,
    ) -> Self {
        geometry.validate();
        let rows = (0..geometry.depth)
            .map(|_| PackedArray::<BITS>::new(geometry.width))
            .collect();
        Self {
            rows: Matrix::from_rows(rows, geometry.width),
            geometry,
            seed_fingerprint,
            hasher,
            update: {
                let _ = update;
                PhantomData
            },
            total_count: 0,
        }
    }

    /// Returns the realized geometry.
    pub const fn geometry(&self) -> CountMinGeometry {
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

    /// Returns the number of counter cells pinned at their maximum value.
    ///
    /// With narrow counters (`BITS` 8 or 16), heavy streams can pin cells
    /// at `2^BITS - 1`. A pinned counter stops counting, so estimates may
    /// **under**estimate — the "never underestimates" guarantee no longer
    /// holds. If this returns non-zero, use wider counters (the default
    /// `BITS = 32` saturates only past four billion) or a wider sketch.
    pub fn saturated_cells(&self) -> usize {
        self.rows
            .iter()
            .map(|row| {
                (0..self.geometry.width)
                    .filter(|&index| row.get(index) == PackedArray::<BITS>::MAX)
                    .count()
            })
            .sum()
    }

    /// Returns whether any counter cell is pinned at its maximum value; see
    /// [`Self::saturated_cells`].
    pub fn is_saturated(&self) -> bool {
        self.saturated_cells() > 0
    }

    fn row_indices(&self, hash: u64) -> impl Iterator<Item = usize> + '_ {
        (0..self.geometry.depth).map(move |row| row_index(hash, row, self.geometry.width))
    }
}

impl<const BITS: u32, S, U> CountMinSketch<BITS, S, U>
where
    S: BuildHasher,
{
    /// Estimates the frequency of `item`.
    pub fn estimate_item<T>(&self, item: &T) -> u64
    where
        T: Hash + ?Sized,
    {
        self.estimate_hash(hash_one(&self.hasher, item))
    }

    /// Estimates the frequency of the item behind a pre-hashed 64-bit
    /// value, using the same derivation as [`Self::estimate_item`]. Useful
    /// when hashes are stored instead of items (e.g. by an
    /// [`crate::sketch::EntropySampler`]).
    pub fn estimate_hash(&self, hash: u64) -> u64 {
        self.row_indices(hash)
            .enumerate()
            .map(|(row, index)| self.rows.row(row).get(index))
            .min()
            .unwrap_or(0)
    }
}

impl<const BITS: u32, S> CountMinSketch<BITS, S, PlainUpdate>
where
    S: BuildHasher,
{
    /// Inserts one occurrence of `item` with the plain update rule.
    pub fn insert_item<T>(&mut self, item: &T)
    where
        T: Hash + ?Sized,
    {
        self.insert_count(item, 1);
    }

    /// Inserts `item` with weight `count`: each addressed cell increases by
    /// `count` (saturating). For byte-weighted frequency estimation — the
    /// standard traffic-measurement use — instead of looping
    /// [`Self::insert_item`].
    pub fn insert_count<T>(&mut self, item: &T, count: u64)
    where
        T: Hash + ?Sized,
    {
        let hash = hash_one(&self.hasher, item);
        let (depth, width) = (self.geometry.depth, self.geometry.width);
        for (row, index) in (0..depth).map(|row| (row, row_index(hash, row, width))) {
            let current = self.rows.row(row).get(index);
            self.rows.row_mut(row).set(
                index,
                current.saturating_add(count).min(PackedArray::<BITS>::MAX),
            );
        }
        self.total_count = self.total_count.saturating_add(count);
    }
}

impl<const BITS: u32, S> CountMinSketch<BITS, S, ConservativeUpdate>
where
    S: BuildHasher,
{
    /// Inserts one occurrence of `item` with conservative update.
    pub fn insert_item<T>(&mut self, item: &T)
    where
        T: Hash + ?Sized,
    {
        self.insert_count(item, 1);
    }

    /// Inserts `item` with weight `count` using the conservative update
    /// rule generalized to weights: every cell rises to at least
    /// `minimum + count`. Cells exactly at the minimum gain the full
    /// weight (the classic rule); cells above it rise too when `count`
    /// exceeds their lead, which is what keeps the never-underestimates
    /// guarantee true for weights above 1. At `count = 1` this reduces to
    /// the classic conservative update.
    ///
    /// Note the naive alternative — adding the whole weight only to cells
    /// at the minimum — can underestimate whenever `count` exceeds a
    /// cell's lead over the minimum, so it is deliberately not used.
    pub fn insert_count<T>(&mut self, item: &T, count: u64)
    where
        T: Hash + ?Sized,
    {
        let hash = hash_one(&self.hasher, item);
        let minimum = self
            .row_indices(hash)
            .enumerate()
            .map(|(row, index)| self.rows.row(row).get(index))
            .min()
            .unwrap_or(0);
        let target = minimum.saturating_add(count).min(PackedArray::<BITS>::MAX);

        // Second pass over a fresh (allocation-free) index iterator.
        let (depth, width) = (self.geometry.depth, self.geometry.width);
        for (row, index) in (0..depth).map(|row| (row, row_index(hash, row, width))) {
            let current = self.rows.row(row).get(index);
            if current < target {
                self.rows.row_mut(row).set(index, target);
            }
        }
        self.total_count = self.total_count.saturating_add(count);
    }
}

impl<T, const BITS: u32, S> crate::traits::Estimator<T> for CountMinSketch<BITS, S, PlainUpdate>
where
    T: Hash + ?Sized,
    S: BuildHasher,
{
    fn estimate(&self, item: &T) -> u64 {
        self.estimate_item(item)
    }

    fn insert_count(&mut self, item: &T, count: u64) {
        self.insert_count(item, count);
    }

    fn total(&self) -> u64 {
        self.total_count()
    }
}

impl<T, const BITS: u32, S> crate::traits::Estimator<T>
    for CountMinSketch<BITS, S, ConservativeUpdate>
where
    T: Hash + ?Sized,
    S: BuildHasher,
{
    fn estimate(&self, item: &T) -> u64 {
        self.estimate_item(item)
    }

    fn insert_count(&mut self, item: &T, count: u64) {
        self.insert_count(item, count);
    }

    fn total(&self) -> u64 {
        self.total_count()
    }
}

impl CountMinGeometry {
    /// Computes Count-Min geometry from target error parameters.
    ///
    /// Uses `width = ceil(e / epsilon)` and `depth = ceil(ln(1 / delta))`.
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
        let width = float::ceil(core::f64::consts::E / epsilon);
        let depth = float::ceil(float::ln(1.0 / delta));
        assert!(
            width <= usize::MAX as f64 && depth <= usize::MAX as f64,
            "computed Count-Min geometry does not fit in usize"
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
        assert!(self.width > 0, "Count-Min width must be greater than zero");
        assert!(self.depth > 0, "Count-Min depth must be greater than zero");
    }
}

impl<const BITS: u32, S, U> Sketch for CountMinSketch<BITS, S, U> {
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

impl<T, const BITS: u32, S> Insert<T> for CountMinSketch<BITS, S, PlainUpdate>
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

impl<T, const BITS: u32, S> Insert<T> for CountMinSketch<BITS, S, ConservativeUpdate>
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

impl<T, const BITS: u32, S, U> EstimateCount<T> for CountMinSketch<BITS, S, U>
where
    T: Hash + ?Sized,
    S: BuildHasher,
{
    fn estimate(&self, item: &T) -> u64 {
        self.estimate_item(item)
    }
}

impl<const BITS: u32, S, U> Merge for CountMinSketch<BITS, S, U> {
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
                    a.saturating_add(b).min(PackedArray::<BITS>::MAX)
                });
        }
        self.total_count = self.total_count.saturating_add(other.total_count);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{CountMinGeometry, CountMinSketch};
    use crate::policy::PlainUpdate;
    use crate::traits::{EstimateCount, Insert, Merge};

    #[test]
    fn plain_count_min_never_underestimates_inserted_counts() {
        let mut sketch = CountMinSketch::with_error(0.01, 0.01);
        for _ in 0..10 {
            sketch.insert_item("a");
        }
        for _ in 0..3 {
            sketch.insert_item("b");
        }

        assert!(sketch.estimate_item("a") >= 10);
        assert!(sketch.estimate_item("b") >= 3);
        assert_eq!(sketch.total_count(), 13);
    }

    #[test]
    fn conservative_update_uses_same_query_capability() {
        let mut sketch = CountMinSketch::conservative_with_error(0.01, 0.01);
        Insert::<str>::insert(&mut sketch, "a").unwrap();
        Insert::<str>::insert(&mut sketch, "a").unwrap();

        assert!(EstimateCount::<str>::estimate(&sketch, "a") >= 2);
    }

    #[test]
    fn narrow_counters_report_saturation() {
        let mut sketch = CountMinSketch::<8>::from_parts(
            CountMinGeometry {
                width: 128,
                depth: 4,
            },
            0,
            crate::hash::DefaultBuildHasher::new(0),
            PlainUpdate,
        );
        assert!(!sketch.is_saturated());
        assert_eq!(sketch.saturated_cells(), 0);

        for _ in 0..300 {
            sketch.insert_item("hot");
        }
        // 8-bit counters pin at 255: all four of the item's row counters.
        assert!(sketch.is_saturated());
        assert_eq!(sketch.saturated_cells(), 4);
        // And the guarantee is visibly broken: the estimate is below the
        // true count of 300.
        assert_eq!(sketch.estimate_item("hot"), 255);
    }

    #[test]
    fn weighted_insert_matches_iterated_inserts_for_plain_update() {
        let mut weighted = CountMinSketch::with_error(0.001, 0.01);
        let mut iterated = CountMinSketch::with_error(0.001, 0.01);
        weighted.insert_count("a", 7);
        for _ in 0..7 {
            iterated.insert_item("a");
        }
        assert_eq!(weighted.estimate_item("a"), iterated.estimate_item("a"));
        assert_eq!(weighted.total_count(), 7);
        assert_eq!(weighted.total_count(), iterated.total_count());
    }

    #[test]
    fn conservative_weighted_insert_adds_weight_once() {
        let mut sketch = CountMinSketch::conservative_with_error(0.001, 0.01);
        sketch.insert_count("a", 5);
        // Byte-weighted conservative update: min cells jump to min + 5 in
        // one step (not min + 1 five times).
        assert!(sketch.estimate_item("a") >= 5);
        sketch.insert_count("a", 3);
        assert!(sketch.estimate_item("a") >= 8);
        assert_eq!(sketch.total_count(), 8);
    }

    #[test]
    fn merge_combines_rows() {
        let geometry = CountMinGeometry {
            width: 128,
            depth: 4,
        };
        let mut left = CountMinSketch::from_geometry(geometry, 1);
        let mut right = CountMinSketch::from_geometry(geometry, 1);
        left.insert_item("a");
        right.insert_item("a");

        left.merge_from(&right).unwrap();
        assert!(left.estimate_item("a") >= 2);
    }
}
