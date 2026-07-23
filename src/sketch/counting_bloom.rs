//! Counting Bloom filter implementation.

use core::convert::Infallible;
use core::hash::{BuildHasher, Hash};

use alloc::vec::Vec;

use crate::block::PackedArray;
use crate::error::MergeError;
#[cfg(any(feature = "std", feature = "libm"))]
use crate::float;
use crate::hash::{DefaultBuildHasher, DoubleHashing, IndexScheme, hash_one};
use crate::policy::{CounterPolicy, Saturating};
#[cfg(any(feature = "std", feature = "libm"))]
use crate::traits::EstimateCardinality;
use crate::traits::{Contains, Insert, Merge, Remove, Sketch};

use super::BloomGeometry;

/// A Bloom filter variant that stores small counters instead of bits.
///
/// Counting Bloom filters support deletion by incrementing a counter for each
/// derived index on insert and decrementing those counters on remove. A query
/// returns maybe-present only when every addressed counter is non-zero.
///
/// ```text
/// insert("x") with k = 3
///
///   hash("x") -> i0, i1, i2
///                |   |   |
/// counters:   [0 2 0 1 0 0 1 0 ...]
///              +   +     +
///
/// remove("x")
///      |
///      v
/// decrement the same counters, unless a counter is saturated/sticky
/// ```
///
/// The default policy is [`Saturating`]. For narrow counters this matters: once
/// a cell reaches its maximum value it remains sticky under deletion, avoiding
/// false negatives that would otherwise be introduced by overflowing counts.
///
/// # References
///
/// - Li Fan, Pei Cao, Jussara Almeida, and Andrei Z. Broder, "Summary Cache:
///   A Scalable Wide-Area Web Cache Sharing Protocol", IEEE/ACM Transactions
///   on Networking, 2000. <https://doi.org/10.1109/90.851975>
/// - Burton H. Bloom, "Space/time trade-offs in hash coding with allowable
///   errors", Communications of the ACM, 1970. <https://doi.org/10.1145/362686.362692>
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CountingBloomFilter<
    const BITS: u32 = 4,
    S = DefaultBuildHasher,
    I = DoubleHashing,
    P = Saturating,
> {
    counters: PackedArray<BITS>,
    geometry: BloomGeometry,
    seed_fingerprint: u64,
    hasher: S,
    scheme: I,
    policy: P,
    inserted: u64,
    expected_items: u64,
}

impl CountingBloomFilter<4, DefaultBuildHasher, DoubleHashing, Saturating> {
    /// Creates a counting Bloom filter with explicit geometry and seed.
    ///
    /// # Panics
    ///
    /// Panics if `geometry` has zero bits or zero hashes.
    pub fn from_geometry(geometry: BloomGeometry, seed: u64) -> Self {
        let hasher = DefaultBuildHasher::new(seed);
        Self::from_parts(
            geometry,
            hasher.seed_fingerprint(),
            hasher,
            DoubleHashing,
            Saturating,
        )
    }

    /// Creates a counting Bloom filter for an expected count and target FPP.
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

    /// Creates a target-sized counting Bloom filter with an explicit seed.
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
        let mut filter = Self::from_geometry(
            BloomGeometry::for_capacity(expected_items, false_positive_rate),
            seed,
        );
        filter.expected_items = expected_items;
        filter
    }
}

impl<const BITS: u32, S, I, P> CountingBloomFilter<BITS, S, I, P> {
    /// Creates a counting Bloom filter from explicit components.
    ///
    /// # Panics
    ///
    /// Panics if `geometry` has zero bits or zero hashes, or if `BITS` is not
    /// in `1..=64`.
    pub fn from_parts(
        geometry: BloomGeometry,
        seed_fingerprint: u64,
        hasher: S,
        scheme: I,
        policy: P,
    ) -> Self {
        geometry.validate();
        Self {
            counters: PackedArray::new(geometry.bits),
            geometry,
            seed_fingerprint,
            hasher,
            scheme,
            policy,
            inserted: 0,
            expected_items: 0,
        }
    }

    /// Returns the realized geometry.
    pub const fn geometry(&self) -> BloomGeometry {
        self.geometry
    }

    /// Returns the seed fingerprint used by merge compatibility checks.
    pub const fn seed_fingerprint(&self) -> u64 {
        self.seed_fingerprint
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

    /// Returns the fraction of counters that are non-zero.
    pub fn fill_ratio(&self) -> f64 {
        self.nonzero_counters() as f64 / self.geometry.bits as f64
    }

    /// Estimates the number of distinct inserted items from non-zero counters.
    ///
    /// Available with the `std` or `libm` feature.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn cardinality(&self) -> f64 {
        let nonzero = self.nonzero_counters();
        if nonzero == 0 {
            return 0.0;
        }
        if nonzero >= self.geometry.bits {
            return f64::INFINITY;
        }
        let m = self.geometry.bits as f64;
        let k = self.geometry.hashes as f64;
        -(m / k) * float::ln(1.0 - nonzero as f64 / m)
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

    /// Returns the byte length of the counter storage.
    pub fn storage_bytes(&self) -> usize {
        self.counters.storage_bytes()
    }

    /// Clears every counter and the insert counter.
    pub fn clear(&mut self) {
        self.counters.clear();
        self.inserted = 0;
    }

    fn nonzero_counters(&self) -> usize {
        (0..self.geometry.bits)
            .filter(|&index| self.counters.get(index) != 0)
            .count()
    }
}

impl<const BITS: u32, S, I, P> CountingBloomFilter<BITS, S, I, P>
where
    S: BuildHasher,
    I: IndexScheme,
    P: CounterPolicy,
{
    /// Inserts `item`, incrementing each addressed counter.
    pub fn insert_item<T>(&mut self, item: &T)
    where
        T: Hash + ?Sized,
    {
        let hash = hash_one(&self.hasher, item);
        for index in self
            .scheme
            .indices(hash, self.geometry.hashes, self.geometry.bits)
        {
            let next = self
                .policy
                .increment(self.counters.get(index), PackedArray::<BITS>::MAX, 1);
            self.counters.set(index, next);
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
            .all(|index| self.counters.get(index) != 0)
    }

    /// Removes one occurrence of `item` when it may be present.
    ///
    /// Returns `false` without modifying the filter if any addressed counter is
    /// zero.
    pub fn remove_item<T>(&mut self, item: &T) -> bool
    where
        T: Hash + ?Sized,
    {
        let hash = hash_one(&self.hasher, item);
        let indices: Vec<_> = self
            .scheme
            .indices(hash, self.geometry.hashes, self.geometry.bits)
            .collect();
        if indices.iter().any(|&index| self.counters.get(index) == 0) {
            return false;
        }
        for index in indices {
            let next = self
                .policy
                .decrement(self.counters.get(index), PackedArray::<BITS>::MAX, 1);
            self.counters.set(index, next);
        }
        true
    }
}

impl<const BITS: u32, S, I, P> Sketch for CountingBloomFilter<BITS, S, I, P> {
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

impl<T, const BITS: u32, S, I, P> Insert<T> for CountingBloomFilter<BITS, S, I, P>
where
    T: Hash + ?Sized,
    S: BuildHasher,
    I: IndexScheme,
    P: CounterPolicy,
{
    type Err = Infallible;

    fn insert(&mut self, item: &T) -> Result<(), Self::Err> {
        self.insert_item(item);
        Ok(())
    }
}

impl<T, const BITS: u32, S, I, P> Contains<T> for CountingBloomFilter<BITS, S, I, P>
where
    T: Hash + ?Sized,
    S: BuildHasher,
    I: IndexScheme,
    P: CounterPolicy,
{
    fn contains(&self, item: &T) -> bool {
        self.contains_item(item)
    }
}

impl<T, const BITS: u32, S, I, P> Remove<T> for CountingBloomFilter<BITS, S, I, P>
where
    T: Hash + ?Sized,
    S: BuildHasher,
    I: IndexScheme,
    P: CounterPolicy,
{
    fn remove(&mut self, item: &T) -> bool {
        self.remove_item(item)
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<const BITS: u32, S, I, P> EstimateCardinality for CountingBloomFilter<BITS, S, I, P> {
    fn cardinality(&self) -> f64 {
        self.cardinality()
    }
}

impl<const BITS: u32, S, I, P> Merge for CountingBloomFilter<BITS, S, I, P>
where
    P: CounterPolicy,
{
    fn merge_from(&mut self, other: &Self) -> Result<(), MergeError> {
        if self.geometry != other.geometry {
            return Err(MergeError::GeometryMismatch);
        }
        if self.seed_fingerprint != other.seed_fingerprint {
            return Err(MergeError::SeedMismatch);
        }
        self.counters.merge_with(&other.counters, |a, b| {
            self.policy.merge(a, b, PackedArray::<BITS>::MAX)
        });
        self.inserted = self.inserted.saturating_add(other.inserted);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::CountingBloomFilter;
    use crate::sketch::BloomGeometry;
    use crate::traits::{Contains, Insert, Merge, Remove};

    #[test]
    fn insert_remove_and_contains() {
        let mut filter = CountingBloomFilter::with_capacity(100, 0.01);
        filter.insert_item("alice");

        assert!(filter.contains_item("alice"));
        assert!(filter.remove_item("alice"));
        assert!(!filter.contains_item("alice"));
        assert!(!filter.remove_item("alice"));
    }

    #[test]
    fn capacity_tracking_reports_fill() {
        let mut filter = CountingBloomFilter::with_capacity(200, 0.01);
        assert_eq!(filter.capacity(), Some(200));
        for i in 0..100_u64 {
            filter.insert_item(&i);
        }
        assert_eq!(filter.inserted_count(), 100);
        assert_eq!(filter.estimated_fill(), Some(0.5));
        filter.clear();
        assert_eq!(filter.estimated_fill(), Some(0.0));
    }

    #[test]
    fn capability_traits_work() {
        let mut filter = CountingBloomFilter::with_capacity(100, 0.01);
        Insert::<str>::insert(&mut filter, "alice").unwrap();

        assert!(Contains::<str>::contains(&filter, "alice"));
        assert!(Remove::<str>::remove(&mut filter, "alice"));
    }

    #[test]
    fn merge_combines_counts() {
        let geometry = BloomGeometry {
            bits: 512,
            hashes: 4,
        };
        let mut left = CountingBloomFilter::from_geometry(geometry, 7);
        let mut right = CountingBloomFilter::from_geometry(geometry, 7);

        left.insert_item("a");
        right.insert_item("b");
        left.merge_from(&right).unwrap();

        assert!(left.contains_item("a"));
        assert!(left.contains_item("b"));
    }
}
