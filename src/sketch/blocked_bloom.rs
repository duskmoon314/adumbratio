//! Blocked Bloom filter implementation.

use core::convert::Infallible;
use core::hash::{BuildHasher, Hash};

use crate::error::MergeError;
use crate::hash::{Blocked, DefaultBuildHasher};
#[cfg(any(feature = "std", feature = "libm"))]
use crate::traits::EstimateCardinality;
use crate::traits::{Contains, Insert, Merge, Sketch};

use super::{BloomFilter, BloomGeometry};

/// A cache-friendly Bloom filter that keeps all probed bits in one block.
///
/// A classical Bloom filter scatters its `k` probed bits across the whole
/// array, so every lookup can miss `k` cache lines. A blocked Bloom filter
/// first selects one contiguous block of bits — one cache line by default —
/// and derives all `k` indices inside it. Lookups and insertions then touch
/// a single cache line, trading a slightly higher false-positive rate for
/// much better memory locality on large filters.
///
/// ```text
/// insert("x")
///      |
///      v
///   hash("x") -> block b, in-block indices j0, j1, j2
///                |
/// blocks:     [ .... | .... | j0 j1 j2 | .... ]
///                            one cache line
/// ```
///
/// This sketch is deliberately thin: it is a [`BloomFilter`] whose index
/// scheme is [`Blocked`], which is exactly the composition the block layer
/// is designed for. Rebuilding it from raw blocks takes a dozen lines:
///
/// ```
/// use adumbratio::block::BitArray;
/// use adumbratio::hash::{Blocked, DefaultBuildHasher, IndexScheme, hash_one};
///
/// let mut bits = BitArray::new(1 << 16);
/// let hasher = DefaultBuildHasher::new(0);
/// let scheme = Blocked::default();
///
/// let hash = hash_one(&hasher, "alice");
/// for index in scheme.indices(hash, 7, bits.len()) {
///     bits.set(index);
/// }
/// let query = hash_one(&hasher, "alice");
/// assert!(scheme.indices(query, 7, bits.len()).all(|i| bits.get(i)));
/// ```
///
/// The standard `m`/`k` sizing formulas are reused from the classical Bloom
/// filter. Because blocking distributes bits non-uniformly, the realized
/// false-positive rate is somewhat higher than the classical prediction;
/// [`Self::expected_fpp`] therefore reports the classical approximation.
///
/// # References
///
/// - Felix Putze, Peter Sanders, and Johannes Singler, "Cache-, Hash-, and
///   Space-Efficient Bloom Filters", ACM Journal of Experimental
///   Algorithmics. <https://doi.org/10.1145/1498698.1594230>
/// - Burton H. Bloom, "Space/time trade-offs in hash coding with allowable
///   errors", Communications of the ACM, 1970. <https://doi.org/10.1145/362686.362692>
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BlockedBloomFilter<S = DefaultBuildHasher> {
    inner: BloomFilter<S, Blocked>,
}

impl BlockedBloomFilter<DefaultBuildHasher> {
    /// Creates a blocked Bloom filter with explicit geometry and seed, using
    /// the default cache-line block size.
    ///
    /// # Panics
    ///
    /// Panics if `geometry` has zero bits or zero hashes.
    pub fn from_geometry(geometry: BloomGeometry, seed: u64) -> Self {
        Self::from_geometry_with_block(geometry, Blocked::CACHE_LINE_BITS, seed)
    }

    /// Creates a blocked Bloom filter with an explicit block size.
    ///
    /// # Panics
    ///
    /// Panics if `geometry` has zero bits or zero hashes, or if `block_bits`
    /// is zero.
    pub fn from_geometry_with_block(geometry: BloomGeometry, block_bits: usize, seed: u64) -> Self {
        let hasher = DefaultBuildHasher::new(seed);
        let inner = BloomFilter::from_parts(
            geometry,
            hasher.seed_fingerprint(),
            hasher,
            Blocked::new(block_bits),
        );
        Self { inner }
    }

    /// Creates a blocked Bloom filter sized for an expected item count and
    /// target FPP, using the default cache-line block size.
    ///
    /// Available with the `std` or `libm` feature.
    ///
    /// # Panics
    ///
    /// Panics if `expected_items` is zero or if `false_positive_rate` is not
    /// in `0.0..1.0`.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn with_capacity(expected_items: u64, false_positive_rate: f64) -> Self {
        Self::with_capacity_and_seed(expected_items, false_positive_rate, 0)
    }

    /// Creates a target-sized blocked Bloom filter with an explicit seed.
    ///
    /// Available with the `std` or `libm` feature.
    ///
    /// # Panics
    ///
    /// Panics if `expected_items` is zero or if `false_positive_rate` is not
    /// in `0.0..1.0`.
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
        filter.inner.mark_capacity(expected_items);
        filter
    }
}

impl<S> BlockedBloomFilter<S> {
    /// Returns the realized geometry.
    pub const fn geometry(&self) -> BloomGeometry {
        self.inner.geometry()
    }

    /// Returns the seed fingerprint used by merge compatibility checks.
    pub const fn seed_fingerprint(&self) -> u64 {
        self.inner.seed_fingerprint()
    }

    /// Returns the block size in bits.
    pub fn block_bits(&self) -> usize {
        self.inner.scheme().block_bits()
    }

    /// Returns the fraction of bits currently set.
    pub fn fill_ratio(&self) -> f64 {
        self.inner.fill_ratio()
    }

    /// Returns the exact number of insert events (not distinct items —
    /// duplicates count separately).
    pub const fn inserted_count(&self) -> u64 {
        self.inner.inserted_count()
    }

    /// Returns the item count the filter was sized for by `with_capacity*`,
    /// or `None` when built from explicit geometry.
    pub const fn capacity(&self) -> Option<u64> {
        self.inner.capacity()
    }

    /// Returns the fraction of the declared capacity used by insert events,
    /// or `None` when no capacity was declared.
    pub fn estimated_fill(&self) -> Option<f64> {
        self.inner.estimated_fill()
    }

    /// Estimates the number of inserted distinct items from the fill ratio.
    ///
    /// Uses the classical Bloom estimate; blocking makes it approximate.
    ///
    /// Available with the `std` or `libm` feature.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn cardinality(&self) -> f64 {
        self.inner.cardinality()
    }

    /// Returns the classical Bloom false-positive probability after `items`
    /// inserts. The blocked realization is somewhat higher.
    ///
    /// Available with the `std` or `libm` feature.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn expected_fpp(&self, items: u64) -> f64 {
        self.inner.expected_fpp(items)
    }

    /// Returns the byte length of the bit storage.
    pub fn storage_bytes(&self) -> usize {
        self.inner.storage_bytes()
    }

    /// Clears all bits.
    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

impl<S> BlockedBloomFilter<S>
where
    S: BuildHasher,
{
    /// Inserts `item`.
    pub fn insert_item<T>(&mut self, item: &T)
    where
        T: Hash + ?Sized,
    {
        self.inner.insert_item(item);
    }

    /// Returns whether `item` may be present.
    pub fn contains_item<T>(&self, item: &T) -> bool
    where
        T: Hash + ?Sized,
    {
        self.inner.contains_item(item)
    }
}

impl<S> Sketch for BlockedBloomFilter<S> {
    fn clear(&mut self) {
        self.clear();
    }

    fn len_hint(&self) -> Option<u64> {
        Sketch::len_hint(&self.inner)
    }

    fn storage_bytes(&self) -> usize {
        self.storage_bytes()
    }
}

impl<T, S> Insert<T> for BlockedBloomFilter<S>
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

impl<T, S> Contains<T> for BlockedBloomFilter<S>
where
    T: Hash + ?Sized,
    S: BuildHasher,
{
    fn contains(&self, item: &T) -> bool {
        self.contains_item(item)
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<S> EstimateCardinality for BlockedBloomFilter<S> {
    fn cardinality(&self) -> f64 {
        self.cardinality()
    }
}

impl<S> Merge for BlockedBloomFilter<S> {
    fn merge_from(&mut self, other: &Self) -> Result<(), MergeError> {
        // The index scheme is part of the layout: merging filters with
        // different block sizes would corrupt answers, and the inner merge
        // check cannot see it.
        if self.block_bits() != other.block_bits() {
            return Err(MergeError::GeometryMismatch);
        }
        self.inner.merge_from(&other.inner)
    }
}

#[cfg(test)]
mod tests {
    use super::BlockedBloomFilter;
    use crate::error::MergeError;
    use crate::sketch::BloomGeometry;
    use crate::traits::{Contains, Insert, Merge};

    #[test]
    fn blocked_capacity_delegates() {
        let mut filter = BlockedBloomFilter::with_capacity(500, 0.01);
        assert_eq!(filter.capacity(), Some(500));
        for i in 0..100_u64 {
            filter.insert_item(&i);
        }
        assert_eq!(filter.inserted_count(), 100);
        assert_eq!(filter.estimated_fill(), Some(0.2));
    }

    #[test]
    fn inserted_items_have_no_false_negatives() {
        let mut filter = BlockedBloomFilter::with_capacity_and_seed(1_000, 0.01, 99);

        for i in 0..1_000_u64 {
            filter.insert_item(&i);
        }

        for i in 0..1_000_u64 {
            assert!(filter.contains_item(&i), "missing inserted item {i}");
        }
    }

    #[test]
    fn capability_traits_work() {
        let mut filter = BlockedBloomFilter::with_capacity(100, 0.01);
        Insert::<str>::insert(&mut filter, "alice").unwrap();

        assert!(Contains::<str>::contains(&filter, "alice"));
        assert_eq!(filter.block_bits(), 512);
    }

    #[test]
    fn merge_matches_inserting_both_streams() {
        let geometry = BloomGeometry {
            bits: 4_096,
            hashes: 5,
        };
        let mut left = BlockedBloomFilter::from_geometry(geometry, 42);
        let mut right = BlockedBloomFilter::from_geometry(geometry, 42);
        let mut combined = BlockedBloomFilter::from_geometry(geometry, 42);

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
    fn merge_rejects_different_block_sizes() {
        let geometry = BloomGeometry {
            bits: 4_096,
            hashes: 5,
        };
        let mut left = BlockedBloomFilter::from_geometry_with_block(geometry, 512, 1);
        let right = BlockedBloomFilter::from_geometry_with_block(geometry, 256, 1);

        assert_eq!(left.merge_from(&right), Err(MergeError::GeometryMismatch));
    }
}
