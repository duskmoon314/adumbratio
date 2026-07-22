//! Cuckoo filter implementation.

use core::hash::{BuildHasher, Hash};

use crate::block::{BucketArray, Fingerprint};
use crate::error::SketchFull;
#[cfg(any(feature = "std", feature = "libm"))]
use crate::float;
use crate::hash::{DefaultBuildHasher, PartialKeyCuckoo, hash_one};
use crate::policy::{KickLoop, XorShift64};
use crate::traits::{Contains, Insert, Remove, Sketch};

/// Explicit Cuckoo filter geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CuckooGeometry {
    /// Number of buckets. Must be a power of two.
    pub buckets: usize,
    /// Number of fingerprint slots in each bucket.
    pub slots_per_bucket: usize,
    /// Number of fingerprint bits stored per item. Must equal the width of
    /// the filter's fingerprint type `F`.
    pub fingerprint_bits: u32,
    /// Maximum number of evictions attempted by one insertion.
    pub max_kicks: usize,
}

/// A Cuckoo filter for approximate membership with deletion.
///
/// A Cuckoo filter stores short fingerprints in buckets. Each item has two
/// candidate buckets: a primary bucket from the item hash and an alternate
/// bucket derived from the primary bucket and fingerprint. The alternate
/// mapping is reversible, so after a fingerprint is kicked out of one bucket
/// the filter can compute its other legal bucket without knowing the original
/// item.
///
/// ```text
/// item "x"
///    |
///    +--> hash -> fingerprint fp
///    +--> hash -> bucket i1
///                  bucket i2 = i1 xor hash(fp)
///
/// buckets:
///   i1 [ fp | .. | .. | .. ]   first choice
///   i2 [ .. | .. | .. | .. ]   alternate choice
///
/// if both buckets are full:
///   place fp by evicting one resident fingerprint
///   move evicted fingerprint to its alternate bucket
///   repeat until an empty slot appears or max_kicks is reached
/// ```
///
/// Unlike a Bloom filter, deletion is supported because removing one
/// fingerprint clears a concrete slot. Insertions are fallible: once the table
/// is too full, the kick loop may be unable to place a displaced fingerprint.
///
/// The fingerprint type `F` fixes the storage width at the type level: the
/// default `u16` uses two bytes per slot regardless of the target FPP. Two
/// items sharing a fingerprint and bucket pair are indistinguishable
/// ("twins"): inserting the second is a no-op, and removing one removes
/// both — only delete items you inserted, as the paper prescribes.
///
/// `Merge` is deliberately not implemented: merging requires re-inserting
/// fingerprints, which can fail at high load, and [`crate::error::MergeError`]
/// cannot express capacity exhaustion.
///
/// # References
///
/// - Bin Fan, Dave G. Andersen, Michael Kaminsky, and Michael D. Mitzenmacher,
///   "Cuckoo Filter: Practically Better Than Bloom", CoNEXT 2014.
///   <https://doi.org/10.1145/2674005.2674994>
/// - Rasmus Pagh and Flemming Friche Rodler, "Cuckoo Hashing", ESA 2001.
///   <https://doi.org/10.1007/3-540-44676-1_10>
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CuckooFilter<F: Fingerprint = u16, S = DefaultBuildHasher> {
    buckets: BucketArray<F>,
    geometry: CuckooGeometry,
    seed_fingerprint: u64,
    hasher: S,
    kick: KickLoop,
    rng: XorShift64,
}

impl CuckooFilter<u16, DefaultBuildHasher> {
    /// Creates a Cuckoo filter from capacity and target FPP.
    ///
    /// Uses four slots per bucket and a 95% target load factor. The stored
    /// fingerprints are `u16`, which reaches target FPPs down to about
    /// 1.3e-4. For a different fingerprint width, build explicit geometry
    /// with [`CuckooGeometry::for_capacity_with_bits`] and call
    /// [`CuckooFilter::from_geometry`] on the chosen `F`.
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

    /// Creates a seeded Cuckoo filter from capacity and target FPP.
    ///
    /// Available with the `std` or `libm` feature.
    ///
    /// # Panics
    ///
    /// Panics under the same conditions as [`Self::with_capacity`].
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn with_capacity_and_seed(
        expected_items: u64,
        false_positive_rate: f64,
        seed: u64,
    ) -> Self {
        Self::from_geometry(
            CuckooGeometry::for_capacity_with_bits(expected_items, false_positive_rate, 16),
            seed,
        )
    }
}

impl<F: Fingerprint> CuckooFilter<F, DefaultBuildHasher> {
    /// Creates a Cuckoo filter with explicit geometry and seed.
    ///
    /// # Panics
    ///
    /// Panics if `geometry` is invalid or if its fingerprint width differs
    /// from `F`'s.
    pub fn from_geometry(geometry: CuckooGeometry, seed: u64) -> Self {
        let hasher = DefaultBuildHasher::new(seed);
        Self::from_parts(
            geometry,
            hasher.seed_fingerprint(),
            hasher,
            XorShift64::new(seed ^ 0xa5a5_5a5a_d3c3_b4b4),
        )
    }
}

impl<F: Fingerprint, S> CuckooFilter<F, S> {
    /// Creates a Cuckoo filter from explicit components.
    ///
    /// # Panics
    ///
    /// Panics if `geometry` is invalid or if its fingerprint width differs
    /// from `F`'s.
    pub fn from_parts(
        geometry: CuckooGeometry,
        seed_fingerprint: u64,
        hasher: S,
        rng: XorShift64,
    ) -> Self {
        geometry.validate();
        assert_eq!(
            geometry.fingerprint_bits,
            F::BITS,
            "geometry fingerprint width {} does not match fingerprint type width {}",
            geometry.fingerprint_bits,
            F::BITS
        );
        Self {
            buckets: BucketArray::new(geometry.buckets, geometry.slots_per_bucket),
            geometry,
            seed_fingerprint,
            hasher,
            kick: KickLoop::new(geometry.max_kicks),
            rng,
        }
    }

    /// Returns the realized geometry.
    pub const fn geometry(&self) -> CuckooGeometry {
        self.geometry
    }

    /// Returns the seed fingerprint used for compatibility checks.
    pub const fn seed_fingerprint(&self) -> u64 {
        self.seed_fingerprint
    }

    /// Returns the number of occupied fingerprint slots.
    pub const fn occupancy(&self) -> usize {
        self.buckets.occupancy()
    }

    /// Returns the occupied fraction of all fingerprint slots.
    pub fn load_factor(&self) -> f64 {
        self.occupancy() as f64 / (self.geometry.buckets * self.geometry.slots_per_bucket) as f64
    }

    /// Returns the approximate false-positive probability for this geometry.
    ///
    /// Available with the `std` or `libm` feature.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn expected_fpp(&self) -> f64 {
        let denominator = float::powi(2.0, self.geometry.fingerprint_bits as i32);
        ((2 * self.geometry.slots_per_bucket) as f64 / denominator).min(1.0)
    }

    /// Returns the byte length of the bucket storage.
    pub fn storage_bytes(&self) -> usize {
        self.buckets.storage_bytes()
    }

    /// Clears all buckets.
    pub fn clear(&mut self) {
        self.buckets.clear();
    }

    fn alternate_bucket(&self, bucket: usize, fingerprint: F) -> usize {
        PartialKeyCuckoo::alt_bucket(bucket, fingerprint.to_u64(), self.geometry.buckets)
    }
}

impl<F: Fingerprint, S> CuckooFilter<F, S>
where
    S: BuildHasher,
{
    /// Inserts `item`.
    ///
    /// If a matching fingerprint already occupies one of the item's buckets,
    /// the insert is deduplicated and succeeds immediately.
    ///
    /// # Errors
    ///
    /// Returns [`SketchFull`] if the bounded kick loop cannot place the final
    /// displaced fingerprint.
    pub fn insert_item<T>(&mut self, item: &T) -> Result<(), SketchFull>
    where
        T: Hash + ?Sized,
    {
        let (fingerprint, first, second) = self.fingerprint_and_buckets(item);
        if self.buckets.contains(first, fingerprint) || self.buckets.contains(second, fingerprint) {
            return Ok(());
        }
        if self.buckets.try_insert(first, fingerprint).is_ok() {
            return Ok(());
        }
        if self.buckets.try_insert(second, fingerprint).is_ok() {
            return Ok(());
        }

        let mut bucket = first;
        let mut fp = self
            .buckets
            .swap_random_slot(bucket, fingerprint, &mut self.rng);
        bucket = self.alternate_bucket(bucket, fp);

        for _ in 0..self.kick.max_kicks {
            if self.buckets.try_insert(bucket, fp).is_ok() {
                return Ok(());
            }
            let evicted = self.buckets.swap_random_slot(bucket, fp, &mut self.rng);
            fp = evicted;
            bucket = self.alternate_bucket(bucket, fp);
        }

        Err(SketchFull::new(fp.to_u64()))
    }

    /// Returns whether `item` may be present.
    pub fn contains_item<T>(&self, item: &T) -> bool
    where
        T: Hash + ?Sized,
    {
        let (fingerprint, first, second) = self.fingerprint_and_buckets(item);
        self.buckets.contains(first, fingerprint) || self.buckets.contains(second, fingerprint)
    }

    /// Removes `item`, returning whether a fingerprint was cleared.
    ///
    /// Removing an item that was never inserted may clear a fingerprint that
    /// belongs to a different item; only remove items known to be present.
    pub fn remove_item<T>(&mut self, item: &T) -> bool
    where
        T: Hash + ?Sized,
    {
        let (fingerprint, first, second) = self.fingerprint_and_buckets(item);
        self.buckets.remove(first, fingerprint) || self.buckets.remove(second, fingerprint)
    }

    fn fingerprint_and_buckets<T>(&self, item: &T) -> (F, usize, usize)
    where
        T: Hash + ?Sized,
    {
        let hash = hash_one(&self.hasher, item);
        let fingerprint = PartialKeyCuckoo::typed_fingerprint::<F>(hash);
        let first = PartialKeyCuckoo::bucket(hash, self.geometry.buckets);
        let second = self.alternate_bucket(first, fingerprint);
        (fingerprint, first, second)
    }
}

impl CuckooGeometry {
    /// Computes Cuckoo geometry from capacity and target FPP, using the
    /// default 16-bit (`u16`) fingerprint width.
    ///
    /// Available with the `std` or `libm` feature.
    ///
    /// # Panics
    ///
    /// Panics under the same conditions as
    /// [`Self::for_capacity_with_bits`].
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn for_capacity(expected_items: u64, false_positive_rate: f64) -> Self {
        Self::for_capacity_with_bits(expected_items, false_positive_rate, 16)
    }

    /// Computes Cuckoo geometry from capacity, target FPP, and fingerprint
    /// width.
    ///
    /// Uses four slots per bucket and rounds bucket count up to a power of
    /// two. The stored width is `fingerprint_bits`; the width only needs to
    /// be large enough that the fingerprint-collision bound
    /// `2 * slots / 2^bits` meets the target FPP.
    ///
    /// Available with the `std` or `libm` feature.
    ///
    /// # Panics
    ///
    /// Panics if `expected_items` is zero, if `false_positive_rate` is not in
    /// `0.0..1.0`, if `fingerprint_bits` is not in `1..=64`, or if the width
    /// cannot reach the target FPP.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn for_capacity_with_bits(
        expected_items: u64,
        false_positive_rate: f64,
        fingerprint_bits: u32,
    ) -> Self {
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
        assert!(
            (1..=64).contains(&fingerprint_bits),
            "fingerprint width must be in 1..=64 bits"
        );
        let slots_per_bucket = 4;
        let achievable =
            (2 * slots_per_bucket) as f64 / float::powi(2.0, fingerprint_bits as i32);
        assert!(
            achievable <= false_positive_rate,
            "fingerprint width {fingerprint_bits} bits cannot reach target false-positive rate {false_positive_rate} with {slots_per_bucket} slots per bucket; use a wider fingerprint type"
        );
        let load = 0.95;
        let needed_buckets = float::ceil(
            (expected_items as f64) / (slots_per_bucket as f64 * load),
        );
        assert!(
            needed_buckets <= usize::MAX as f64,
            "computed Cuckoo bucket count does not fit in usize"
        );
        let buckets = (needed_buckets.max(1.0) as usize).next_power_of_two();
        Self {
            buckets,
            slots_per_bucket,
            fingerprint_bits,
            max_kicks: KickLoop::default().max_kicks,
        }
    }

    /// Validates the geometry.
    ///
    /// # Panics
    ///
    /// Panics if buckets or slots are zero, if buckets is not a power of two,
    /// if fingerprint bits is not in `1..=64`, or if `max_kicks` is zero.
    pub fn validate(self) {
        assert!(
            self.buckets > 0,
            "Cuckoo bucket count must be greater than zero"
        );
        assert!(
            self.buckets.is_power_of_two(),
            "Cuckoo bucket count must be a power of two"
        );
        assert!(
            self.slots_per_bucket > 0,
            "Cuckoo slots per bucket must be greater than zero"
        );
        assert!(
            (1..=64).contains(&self.fingerprint_bits),
            "Cuckoo fingerprint bits must be in 1..=64"
        );
        assert!(
            self.max_kicks > 0,
            "Cuckoo max_kicks must be greater than zero"
        );
    }
}

impl<F: Fingerprint, S> Sketch for CuckooFilter<F, S> {
    fn clear(&mut self) {
        self.clear();
    }

    fn len_hint(&self) -> Option<u64> {
        Some(self.occupancy() as u64)
    }

    fn storage_bytes(&self) -> usize {
        self.storage_bytes()
    }
}

impl<T, F: Fingerprint, S> Insert<T> for CuckooFilter<F, S>
where
    T: Hash + ?Sized,
    S: BuildHasher,
{
    type Err = SketchFull;

    fn insert(&mut self, item: &T) -> Result<(), Self::Err> {
        self.insert_item(item)
    }
}

impl<T, F: Fingerprint, S> Contains<T> for CuckooFilter<F, S>
where
    T: Hash + ?Sized,
    S: BuildHasher,
{
    fn contains(&self, item: &T) -> bool {
        self.contains_item(item)
    }
}

impl<T, F: Fingerprint, S> Remove<T> for CuckooFilter<F, S>
where
    T: Hash + ?Sized,
    S: BuildHasher,
{
    fn remove(&mut self, item: &T) -> bool {
        self.remove_item(item)
    }
}

#[cfg(test)]
mod tests {
    use super::{CuckooFilter, CuckooGeometry};
    use crate::traits::{Contains, Insert, Remove, Sketch};

    #[test]
    fn insert_contains_and_remove() {
        let mut filter = CuckooFilter::with_capacity(128, 0.01);
        filter.insert_item(&42_u64).unwrap();

        assert!(filter.contains_item(&42_u64));
        assert!(filter.remove_item(&42_u64));
        assert!(!filter.contains_item(&42_u64));
    }

    #[test]
    fn capability_traits_work() {
        let mut filter = CuckooFilter::with_capacity(128, 0.01);
        Insert::<str>::insert(&mut filter, "alice").unwrap();

        assert!(Contains::<str>::contains(&filter, "alice"));
        assert!(Remove::<str>::remove(&mut filter, "alice"));
        assert_eq!(Sketch::len_hint(&filter), Some(0));
    }

    #[test]
    fn default_fingerprint_is_u16_and_storage_matches() {
        let filter = CuckooFilter::with_capacity(1_000, 0.01);
        let geometry = filter.geometry();
        assert_eq!(geometry.fingerprint_bits, 16);
        assert_eq!(
            filter.storage_bytes(),
            geometry.buckets * geometry.slots_per_bucket * size_of::<u16>()
        );
    }

    #[test]
    fn narrower_fingerprint_types_work() {
        let geometry = CuckooGeometry {
            buckets: 64,
            slots_per_bucket: 4,
            fingerprint_bits: 8,
            max_kicks: 10,
        };
        let mut filter = CuckooFilter::<u8>::from_geometry(geometry, 3);
        filter.insert_item(&7_u64).unwrap();

        assert!(filter.contains_item(&7_u64));
        assert!(filter.remove_item(&7_u64));
        assert!(!filter.contains_item(&7_u64));
    }

    #[test]
    #[should_panic(expected = "does not match fingerprint type width")]
    fn geometry_width_must_match_fingerprint_type() {
        let geometry = CuckooGeometry {
            buckets: 64,
            slots_per_bucket: 4,
            fingerprint_bits: 12,
            max_kicks: 10,
        };
        CuckooFilter::<u8>::from_geometry(geometry, 0);
    }

    #[test]
    #[should_panic(expected = "cannot reach target false-positive rate")]
    fn narrow_fingerprint_rejects_tight_fpp() {
        // 8-bit fingerprints cannot reach below 8/2^8 = 0.03125.
        CuckooGeometry::for_capacity_with_bits(128, 0.01, 8);
    }

    #[test]
    fn tiny_filter_can_report_full() {
        let geometry = CuckooGeometry {
            buckets: 1,
            slots_per_bucket: 1,
            fingerprint_bits: 16,
            max_kicks: 1,
        };
        let mut filter: CuckooFilter = CuckooFilter::from_geometry(geometry, 0);
        filter.insert_item(&0_u64).unwrap();

        let mut saw_full = false;
        for item in 1..100_u64 {
            if filter.insert_item(&item).is_err() {
                saw_full = true;
                break;
            }
        }
        assert!(saw_full);
    }
}
