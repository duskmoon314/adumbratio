//! Bloom filter implementation.

use core::convert::Infallible;
use core::hash::{BuildHasher, Hash};

use crate::block::BitArray;
use crate::error::MergeError;
#[cfg(any(feature = "std", feature = "libm"))]
use crate::float;
use crate::hash::{DefaultBuildHasher, DoubleHashing, IndexScheme, hash_one};
#[cfg(any(feature = "std", feature = "libm"))]
use crate::traits::EstimateCardinality;
use crate::traits::{Contains, Insert, Merge, Sketch};

/// Explicit Bloom filter geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BloomGeometry {
    /// Number of bits in the filter.
    pub bits: usize,
    /// Number of hash-derived indices set for each inserted item.
    pub hashes: usize,
}

/// A Bloom filter for approximate set membership.
///
/// A Bloom filter stores only a bit array. To insert an item, the item is
/// hashed once, expanded into `k` indices, and each addressed bit is set. To
/// query, the same indices are checked. If any bit is zero, the item is
/// definitely absent. If every bit is one, the item may be present.
///
/// ```text
/// insert("x")
///      |
///      v
///   hash("x") -> i0, i1, i2
///                |   |   |
/// bits:       [0 1 0 1 0 0 1 0 ...]
///
/// contains("x")
///      |
///      v
///   same indices -> all bits set? yes => maybe present
///                   any bit clear? no  => definitely absent
/// ```
///
/// This implementation uses [`BitArray`](crate::block::BitArray) for storage
/// and [`DoubleHashing`](crate::hash::DoubleHashing) by default for index
/// expansion.
///
/// # References
///
/// - Burton H. Bloom, "Space/time trade-offs in hash coding with allowable
///   errors", Communications of the ACM, 1970. <https://doi.org/10.1145/362686.362692>
/// - Adam Kirsch and Michael Mitzenmacher, "Less Hashing, Same Performance:
///   Building a Better Bloom Filter", ESA 2006. <https://doi.org/10.1007/11841036_21>
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BloomFilter<S = DefaultBuildHasher, I = DoubleHashing> {
    bits: BitArray,
    geometry: BloomGeometry,
    seed_fingerprint: u64,
    hasher: S,
    scheme: I,
    inserted: u64,
    expected_items: u64,
}

impl BloomFilter<DefaultBuildHasher, DoubleHashing> {
    /// Creates a Bloom filter with explicit geometry and hash seed.
    ///
    /// # Panics
    ///
    /// Panics if `geometry` has zero bits or zero hashes.
    pub fn from_geometry(geometry: BloomGeometry, seed: u64) -> Self {
        let hasher = DefaultBuildHasher::new(seed);
        Self::from_parts(geometry, hasher.seed_fingerprint(), hasher, DoubleHashing)
    }

    /// Creates a Bloom filter sized for an expected item count and target FPP.
    ///
    /// The seed is zero. Use [`Self::with_capacity_and_seed`] when merge
    /// compatibility must be controlled explicitly.
    ///
    /// Available with the `std` or `libm` feature.
    ///
    /// # Panics
    ///
    /// Panics if `expected_items` is zero or if `false_positive_rate` is not in
    /// `0.0..1.0`.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn with_capacity(expected_items: u64, false_positive_rate: f64) -> Self {
        Self::with_capacity_and_seed(expected_items, false_positive_rate, 0)
    }

    /// Creates a target-sized Bloom filter with an explicit hash seed.
    ///
    /// Available with the `std` or `libm` feature.
    ///
    /// # Panics
    ///
    /// Panics if `expected_items` is zero or if `false_positive_rate` is not in
    /// `0.0..1.0`.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn with_capacity_and_seed(
        expected_items: u64,
        false_positive_rate: f64,
        seed: u64,
    ) -> Self {
        let geometry = BloomGeometry::for_capacity(expected_items, false_positive_rate);
        let mut filter = Self::from_geometry(geometry, seed);
        filter.mark_capacity(expected_items);
        filter
    }
}

impl<S, I> BloomFilter<S, I> {
    /// Creates a Bloom filter from explicit components.
    ///
    /// `seed_fingerprint` is compared during merge to prevent combining
    /// filters built from incompatible hash parameters.
    ///
    /// # Panics
    ///
    /// Panics if `geometry` has zero bits or zero hashes.
    pub fn from_parts(
        geometry: BloomGeometry,
        seed_fingerprint: u64,
        hasher: S,
        scheme: I,
    ) -> Self {
        geometry.validate();
        Self {
            bits: BitArray::new(geometry.bits),
            geometry,
            seed_fingerprint,
            hasher,
            scheme,
            inserted: 0,
            expected_items: 0,
        }
    }

    /// Returns the realized geometry.
    pub const fn geometry(&self) -> BloomGeometry {
        self.geometry
    }

    /// Returns the exact number of insert events (not distinct items —
    /// duplicates count separately).
    pub const fn inserted_count(&self) -> u64 {
        self.inserted
    }

    /// Returns the item count the filter was sized for by `with_capacity*`,
    /// or `None` when built from explicit geometry.
    pub const fn capacity(&self) -> Option<u64> {
        if self.expected_items > 0 {
            Some(self.expected_items)
        } else {
            None
        }
    }

    /// Returns the fraction of the declared capacity used by insert events
    /// (`inserted_count / capacity`), or `None` when no capacity was
    /// declared. Consumers use this for bounded admission instead of
    /// tracking inserts themselves.
    pub fn estimated_fill(&self) -> Option<f64> {
        self.capacity()
            .map(|capacity| self.inserted as f64 / capacity as f64)
    }

    /// Returns the seed fingerprint used by merge compatibility checks.
    pub const fn seed_fingerprint(&self) -> u64 {
        self.seed_fingerprint
    }

    /// Returns the index derivation scheme.
    pub const fn scheme(&self) -> &I {
        &self.scheme
    }

    /// Returns the fraction of bits currently set.
    pub fn fill_ratio(&self) -> f64 {
        self.bits.count_ones() as f64 / self.geometry.bits as f64
    }

    /// Estimates the number of inserted distinct items from the fill ratio.
    ///
    /// Available with the `std` or `libm` feature.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn cardinality(&self) -> f64 {
        let set_bits = self.bits.count_ones();
        if set_bits == 0 {
            return 0.0;
        }
        if set_bits >= self.geometry.bits {
            return f64::INFINITY;
        }

        let m = self.geometry.bits as f64;
        let k = self.geometry.hashes as f64;
        -(m / k) * float::ln(1.0 - set_bits as f64 / m)
    }

    /// Returns the expected false-positive probability after `items` inserts.
    ///
    /// Available with the `std` or `libm` feature.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn expected_fpp(&self, items: u64) -> f64 {
        if items == 0 {
            return 0.0;
        }
        let m = self.geometry.bits as f64;
        let k = self.geometry.hashes as f64;
        float::powf(1.0 - float::exp(-(k * items as f64) / m), k)
    }

    /// Returns the byte length of the bit storage.
    pub fn storage_bytes(&self) -> usize {
        self.bits.storage_bytes()
    }

    /// Clears all bits and the insert counter.
    pub fn clear(&mut self) {
        self.bits.clear();
        self.inserted = 0;
    }

    /// Records the declared capacity for [`Self::capacity`] and
    /// [`Self::estimated_fill`] (used by capacity-solving constructors).
    #[cfg(any(feature = "std", feature = "libm"))]
    pub(crate) fn mark_capacity(&mut self, expected_items: u64) {
        self.expected_items = expected_items;
    }
}

impl<S, I> BloomFilter<S, I>
where
    S: BuildHasher,
    I: IndexScheme,
{
    /// Inserts `item`.
    pub fn insert_item<T>(&mut self, item: &T)
    where
        T: Hash + ?Sized,
    {
        let hash = hash_one(&self.hasher, item);
        for index in self
            .scheme
            .indices(hash, self.geometry.hashes, self.geometry.bits)
        {
            self.bits.set(index);
        }
        self.inserted += 1;
    }

    /// Returns whether `item` may be present.
    pub fn contains_item<T>(&self, item: &T) -> bool
    where
        T: Hash + ?Sized,
    {
        let hash = hash_one(&self.hasher, item);
        self.scheme
            .indices(hash, self.geometry.hashes, self.geometry.bits)
            .all(|index| self.bits.get(index))
    }
}

impl BloomGeometry {
    /// Computes the standard Bloom geometry for `expected_items` and FPP.
    ///
    /// Available with the `std` or `libm` feature.
    ///
    /// # Panics
    ///
    /// Panics if `expected_items` is zero, if `false_positive_rate` is not in
    /// `0.0..1.0`, or if the computed bit count does not fit in `usize`.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn for_capacity(expected_items: u64, false_positive_rate: f64) -> Self {
        assert!(
            expected_items > 0,
            "expected item count must be greater than zero"
        );
        assert!(
            false_positive_rate.is_finite()
                && false_positive_rate > 0.0
                && false_positive_rate < 1.0,
            "false-positive rate must be finite and in 0.0..1.0"
        );

        let n = expected_items as f64;
        let ln2 = core::f64::consts::LN_2;
        let bits = float::ceil(-(n * float::ln(false_positive_rate)) / (ln2 * ln2));
        assert!(
            bits.is_finite() && bits <= usize::MAX as f64,
            "computed Bloom filter bit count does not fit in usize"
        );

        let bits = bits.max(1.0) as usize;
        let hashes = float::round((bits as f64 / n) * ln2).max(1.0) as usize;
        Self { bits, hashes }
    }

    /// Validates that both geometry dimensions are non-zero.
    ///
    /// # Panics
    ///
    /// Panics if `bits` or `hashes` is zero.
    pub fn validate(self) {
        assert!(
            self.bits > 0,
            "Bloom filter bit count must be greater than zero"
        );
        assert!(
            self.hashes > 0,
            "Bloom filter hash count must be greater than zero"
        );
    }
}

impl<S, I> Sketch for BloomFilter<S, I> {
    fn clear(&mut self) {
        self.clear();
    }

    fn len_hint(&self) -> Option<u64> {
        #[cfg(any(feature = "std", feature = "libm"))]
        {
            let estimate = self.cardinality();
            if estimate.is_finite() && estimate >= 0.0 && estimate <= u64::MAX as f64 {
                Some(float::round(estimate) as u64)
            } else {
                None
            }
        }
        #[cfg(not(any(feature = "std", feature = "libm")))]
        {
            None
        }
    }

    fn storage_bytes(&self) -> usize {
        self.storage_bytes()
    }
}

impl<T, S, I> Insert<T> for BloomFilter<S, I>
where
    T: Hash + ?Sized,
    S: BuildHasher,
    I: IndexScheme,
{
    type Err = Infallible;

    fn insert(&mut self, item: &T) -> Result<(), Self::Err> {
        self.insert_item(item);
        Ok(())
    }
}

impl<T, S, I> Contains<T> for BloomFilter<S, I>
where
    T: Hash + ?Sized,
    S: BuildHasher,
    I: IndexScheme,
{
    fn contains(&self, item: &T) -> bool {
        self.contains_item(item)
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<S, I> EstimateCardinality for BloomFilter<S, I> {
    fn cardinality(&self) -> f64 {
        self.cardinality()
    }
}

impl<S, I> Merge for BloomFilter<S, I> {
    fn merge_from(&mut self, other: &Self) -> Result<(), MergeError> {
        if self.geometry != other.geometry {
            return Err(MergeError::GeometryMismatch);
        }
        if self.seed_fingerprint != other.seed_fingerprint {
            return Err(MergeError::SeedMismatch);
        }

        self.bits.union_with(&other.bits);
        self.inserted = self.inserted.saturating_add(other.inserted);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{BloomFilter, BloomGeometry};
    use crate::error::MergeError;
    use crate::traits::{Contains, EstimateCardinality, Insert, Merge, Sketch};

    #[test]
    fn capacity_solver_produces_sensible_geometry() {
        let geometry = BloomGeometry::for_capacity(10_000, 0.01);
        assert!(geometry.bits > 10_000);
        assert!(geometry.hashes > 1);

        let filter = BloomFilter::with_capacity(10_000, 0.01);
        assert!(filter.expected_fpp(10_000) <= 0.011);
    }

    #[test]
    #[should_panic(expected = "expected item count must be greater than zero")]
    fn capacity_rejects_zero_items() {
        BloomFilter::with_capacity(0, 0.01);
    }

    #[test]
    #[should_panic(expected = "false-positive rate must be finite and in 0.0..1.0")]
    fn capacity_rejects_invalid_fpp() {
        BloomFilter::with_capacity(10, 1.0);
    }

    #[test]
    #[should_panic(expected = "Bloom filter bit count must be greater than zero")]
    fn geometry_rejects_zero_bits() {
        BloomFilter::from_geometry(BloomGeometry { bits: 0, hashes: 1 }, 0);
    }

    #[test]
    #[should_panic(expected = "Bloom filter hash count must be greater than zero")]
    fn geometry_rejects_zero_hashes() {
        BloomFilter::from_geometry(
            BloomGeometry {
                bits: 64,
                hashes: 0,
            },
            0,
        );
    }

    #[test]
    fn inserted_items_have_no_false_negatives() {
        let mut filter = BloomFilter::with_capacity_and_seed(1_000, 0.01, 99);

        for i in 0..1_000_u64 {
            filter.insert_item(&i);
        }

        for i in 0..1_000_u64 {
            assert!(filter.contains_item(&i), "missing inserted item {i}");
        }
    }

    #[test]
    fn capacity_tracking_reports_fill_and_resets_on_clear() {
        let mut filter = BloomFilter::with_capacity(1_000, 0.01);
        assert_eq!(filter.capacity(), Some(1_000));
        assert_eq!(filter.inserted_count(), 0);
        assert_eq!(filter.estimated_fill(), Some(0.0));

        for i in 0..250_u64 {
            filter.insert_item(&i);
        }
        assert_eq!(filter.inserted_count(), 250);
        assert_eq!(filter.estimated_fill(), Some(0.25));

        filter.clear();
        assert_eq!(filter.inserted_count(), 0);
        assert_eq!(filter.estimated_fill(), Some(0.0));
        assert_eq!(filter.capacity(), Some(1_000));

        let geometry_only = BloomFilter::from_geometry(
            BloomGeometry {
                bits: 1_024,
                hashes: 3,
            },
            0,
        );
        assert_eq!(geometry_only.capacity(), None);
        assert_eq!(geometry_only.estimated_fill(), None);
    }

    #[test]
    fn capability_traits_work_for_unsized_items() {
        let mut filter = BloomFilter::with_capacity(100, 0.01);
        Insert::<str>::insert(&mut filter, "alice").unwrap();

        assert!(Contains::<str>::contains(&filter, "alice"));
    }

    #[test]
    fn cardinality_and_len_hint_are_finite_until_full() {
        let mut filter = BloomFilter::from_geometry(
            BloomGeometry {
                bits: 1_024,
                hashes: 3,
            },
            0,
        );
        assert_eq!(EstimateCardinality::cardinality(&filter), 0.0);
        assert_eq!(Sketch::len_hint(&filter), Some(0));

        for i in 0..100_u64 {
            filter.insert_item(&i);
        }

        let estimate = EstimateCardinality::cardinality(&filter);
        assert!(estimate.is_finite());
        assert!(estimate > 0.0);
        assert!(Sketch::len_hint(&filter).unwrap() > 0);
    }

    #[test]
    fn merge_matches_inserting_both_streams() {
        let geometry = BloomGeometry {
            bits: 4_096,
            hashes: 5,
        };
        let mut left = BloomFilter::from_geometry(geometry, 42);
        let mut right = BloomFilter::from_geometry(geometry, 42);
        let mut combined = BloomFilter::from_geometry(geometry, 42);

        for i in 0..250_u64 {
            left.insert_item(&i);
            combined.insert_item(&i);
        }
        for i in 250..500_u64 {
            right.insert_item(&i);
            combined.insert_item(&i);
        }

        left.merge_from(&right).unwrap();

        for i in 0..500_u64 {
            assert_eq!(left.contains_item(&i), combined.contains_item(&i));
        }
        assert_eq!(left.fill_ratio(), combined.fill_ratio());
    }

    #[test]
    fn merge_rejects_geometry_mismatch() {
        let mut left = BloomFilter::from_geometry(
            BloomGeometry {
                bits: 1_024,
                hashes: 3,
            },
            1,
        );
        let right = BloomFilter::from_geometry(
            BloomGeometry {
                bits: 2_048,
                hashes: 3,
            },
            1,
        );

        assert_eq!(left.merge_from(&right), Err(MergeError::GeometryMismatch));
    }

    #[test]
    fn merge_rejects_seed_mismatch() {
        let geometry = BloomGeometry {
            bits: 1_024,
            hashes: 3,
        };
        let mut left = BloomFilter::from_geometry(geometry, 1);
        let right = BloomFilter::from_geometry(geometry, 2);

        assert_eq!(left.merge_from(&right), Err(MergeError::SeedMismatch));
    }
}
