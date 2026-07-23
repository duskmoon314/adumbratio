//! Semi-sorted cuckoo filter implementation.

use core::hash::{BuildHasher, Hash};

use crate::block::PackedArray;
use crate::error::SketchFull;
use crate::hash::{DefaultBuildHasher, PartialKeyCuckoo, hash_one};
use crate::policy::{KickLoop, XorShift64};
use crate::traits::{Contains, Insert, Remove, Sketch};

use super::CuckooGeometry;

/// Fingerprint width in bits; the four sorted fingerprints plus the
/// sorting savings pack into 56 bits per bucket.
pub const SEMI_SORTED_FINGERPRINT_BITS: u32 = 15;

/// Binomial coefficients for ranks up to 4. The search range is bounded by
/// `2^15 + 2`, where the u64 products still fit ((2^15+2)^4 < 2^60).
fn choose(v: u64, k: u32) -> u64 {
    if v < k as u64 {
        return 0; // C(v, k) = 0 for v < k by convention
    }
    match k {
        0 => 1,
        1 => v,
        2 => v * (v - 1) / 2,
        3 => v * (v - 1) * (v - 2) / 6,
        4 => v * (v - 1) * (v - 2) * (v - 3) / 24,
        _ => unreachable!("binomial rank out of range"),
    }
}

/// Ranks a sorted multiset of four 15-bit values (zeros allowed and sorted
/// first) in the combinatorial number system. The map `z_i = x_i + i`
/// takes multisets to strictly increasing sequences, and
/// `rank = sum C(z_i, i+1)`. The all-zero multiset ranks to 0, so an empty
/// bucket is rank zero.
fn rank4(sorted: [u64; 4]) -> u64 {
    choose(sorted[0], 1)
        + choose(sorted[1] + 1, 2)
        + choose(sorted[2] + 2, 3)
        + choose(sorted[3] + 3, 4)
}

/// Unranks a combinatorial rank back to the sorted multiset of four.
fn unrank4(mut rank: u64) -> [u64; 4] {
    let mut out = [0_u64; 4];
    for i in (0..4_usize).rev() {
        let k = i as u32 + 1;
        // Largest v with C(v, k) <= rank; strictly decreasing below the
        // previous z keeps the sequence strictly increasing.
        let upper = if i == 3 {
            // z_3 = x_3 + 3 with x_3 a 15-bit value.
            (1_u64 << SEMI_SORTED_FINGERPRINT_BITS) + 2
        } else {
            out[i + 1] + (i as u64)
        };
        let mut lo = 0_u64;
        let mut hi = upper;
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if choose(mid, k) <= rank {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        out[i] = lo;
        rank -= choose(lo, k);
        out[i] -= i as u64;
    }
    out
}

/// A semi-sorted cuckoo filter: the cuckoo filter of Fan et al. 2014 with
/// the paper's own space optimization — the four fingerprints in each
/// bucket are stored *sorted*, and the sorted bucket is packed as a
/// combinatorial rank.
///
/// A sorted multiset of four 15-bit values has `C(2^15 + 3, 4)` states,
/// about 55.4 bits of information, so a bucket packs into 56 bits instead
/// of the 60 that four raw 15-bit slots would need (and the 64 that the
/// plain u16 cuckoo filter uses). The false-positive rate is the same
/// `2 * slots / 2^f` at `f = 15` bits.
///
/// ```text
/// bucket (plain):    [ fp15 | fp15 | fp15 | fp15 ]   60 bits
/// bucket (semi):     rank(sorted multiset)           56 bits
/// ```
///
/// The trade is speed for space: bucket reads and writes cost a
/// rank/unrank round-trip instead of a slot access. Everything else —
/// partial-key cuckoo placement, the kick loop, deduplication, the
/// fingerprint-twin caveat — is identical to
/// [`CuckooFilter`](crate::sketch::CuckooFilter), and `Merge` is omitted
/// for the same reason.
///
/// # References
///
/// - Bin Fan, Dave G. Andersen, Michael Kaminsky, and Michael D.
///   Mitzenmacher, "Cuckoo Filter: Practically Better Than Bloom", CoNEXT
///   2014, Section 4.2. <https://doi.org/10.1145/2674005.2674994>
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SemiSortedCuckooFilter<S = DefaultBuildHasher> {
    buckets: PackedArray<56>,
    geometry: CuckooGeometry,
    seed_fingerprint: u64,
    hasher: S,
    kick: KickLoop,
    rng: XorShift64,
}

impl SemiSortedCuckooFilter<DefaultBuildHasher> {
    /// Creates a semi-sorted cuckoo filter from capacity and target FPP.
    ///
    /// The fingerprint width is fixed at 15 bits (FPP about 2.4e-4);
    /// tighter targets are not available in this variant. Uses four slots
    /// per bucket and a 95% target load factor.
    ///
    /// # Panics
    ///
    /// Panics if `expected_items` is zero or if `false_positive_rate` is
    /// not in `0.0..1.0` or is below the 15-bit bound.
    ///
    /// Available with the `std` or `libm` feature.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn with_capacity(expected_items: u64, false_positive_rate: f64) -> Self {
        Self::with_capacity_and_seed(expected_items, false_positive_rate, 0)
    }

    /// Creates a seeded semi-sorted cuckoo filter from capacity and FPP.
    ///
    /// # Panics
    ///
    /// Panics under the same conditions as [`Self::with_capacity`].
    ///
    /// Available with the `std` or `libm` feature.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn with_capacity_and_seed(
        expected_items: u64,
        false_positive_rate: f64,
        seed: u64,
    ) -> Self {
        Self::from_geometry(
            CuckooGeometry::for_capacity_with_bits(
                expected_items,
                false_positive_rate,
                SEMI_SORTED_FINGERPRINT_BITS,
            ),
            seed,
        )
    }

    /// Creates a semi-sorted cuckoo filter with explicit geometry and seed.
    ///
    /// # Panics
    ///
    /// Panics if `geometry` is invalid or its fingerprint width is not 15.
    pub fn from_geometry(geometry: CuckooGeometry, seed: u64) -> Self {
        assert_eq!(
            geometry.fingerprint_bits,
            SEMI_SORTED_FINGERPRINT_BITS,
            "semi-sorted cuckoo filters use 15-bit fingerprints"
        );
        let hasher = DefaultBuildHasher::new(seed);
        Self::from_parts(
            geometry,
            hasher.seed_fingerprint(),
            hasher,
            XorShift64::new(seed ^ 0xa5a5_5a5a_d3c3_b4b4),
        )
    }
}

impl<S> SemiSortedCuckooFilter<S> {
    /// Creates a semi-sorted cuckoo filter from explicit components.
    ///
    /// # Panics
    ///
    /// Panics if `geometry` is invalid or its fingerprint width is not 15.
    pub fn from_parts(
        geometry: CuckooGeometry,
        seed_fingerprint: u64,
        hasher: S,
        rng: XorShift64,
    ) -> Self {
        geometry.validate();
        assert_eq!(
            geometry.fingerprint_bits,
            SEMI_SORTED_FINGERPRINT_BITS,
            "semi-sorted cuckoo filters use 15-bit fingerprints"
        );
        Self {
            buckets: PackedArray::new(geometry.buckets),
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

    /// Returns the number of stored fingerprints.
    pub fn occupancy(&self) -> usize {
        (0..self.geometry.buckets)
            .map(|bucket| {
                unrank4(self.buckets.get(bucket))
                    .iter()
                    .filter(|&&fp| fp != 0)
                    .count()
            })
            .sum()
    }

    /// Returns the occupied fraction of all fingerprint slots.
    pub fn load_factor(&self) -> f64 {
        self.occupancy() as f64 / (self.geometry.buckets * self.geometry.slots_per_bucket) as f64
    }

    /// Returns the approximate false-positive probability, `8 / 2^15`.
    pub fn expected_fpp(&self) -> f64 {
        (2 * self.geometry.slots_per_bucket) as f64
            / (1_u64 << SEMI_SORTED_FINGERPRINT_BITS) as f64
    }

    /// Returns the byte length of the bucket storage: 56 bits per bucket.
    pub fn storage_bytes(&self) -> usize {
        self.buckets.storage_bytes()
    }

    /// Clears all buckets.
    pub fn clear(&mut self) {
        self.buckets.clear();
    }

    fn alternate_bucket(&self, bucket: usize, fingerprint: u64) -> usize {
        PartialKeyCuckoo::alt_bucket(bucket, fingerprint, self.geometry.buckets)
    }

    // -- bucket-level operations on the packed ranks -------------------------

    fn bucket_contains(&self, bucket: usize, fp: u64) -> bool {
        debug_assert!(bucket < self.geometry.buckets);
        unrank4(self.buckets.get(bucket)).contains(&fp)
    }

    fn bucket_insert(&mut self, bucket: usize, fp: u64) -> bool {
        debug_assert!(bucket < self.geometry.buckets && fp != 0);
        let mut slots = unrank4(self.buckets.get(bucket));
        let Some(empty) = slots.iter().position(|&slot| slot == 0) else {
            return false;
        };
        slots[empty] = fp;
        slots.sort_unstable();
        self.buckets.set(bucket, rank4(slots));
        true
    }

    fn bucket_remove(&mut self, bucket: usize, fp: u64) -> bool {
        debug_assert!(bucket < self.geometry.buckets);
        let mut slots = unrank4(self.buckets.get(bucket));
        let Some(position) = slots.iter().position(|&slot| slot == fp) else {
            return false;
        };
        slots[position] = 0;
        slots.sort_unstable();
        self.buckets.set(bucket, rank4(slots));
        true
    }

    fn bucket_swap_random_slot(&mut self, bucket: usize, fp: u64) -> u64 {
        debug_assert!(bucket < self.geometry.buckets && fp != 0);
        let mut slots = unrank4(self.buckets.get(bucket));
        let index = crate::policy::RngLite::next_index(&mut self.rng, 4);
        let evicted = slots[index];
        slots[index] = fp;
        slots.sort_unstable();
        self.buckets.set(bucket, rank4(slots));
        debug_assert_ne!(evicted, 0, "swap requires a full bucket");
        evicted
    }
}

impl<S> SemiSortedCuckooFilter<S>
where
    S: BuildHasher,
{
    /// Inserts `item`.
    ///
    /// # Errors
    ///
    /// Returns [`SketchFull`] if the bounded kick loop cannot place the
    /// final displaced fingerprint.
    pub fn insert_item<T>(&mut self, item: &T) -> Result<(), SketchFull>
    where
        T: Hash + ?Sized,
    {
        let (fingerprint, first, second) = self.fingerprint_and_buckets(item);
        if self.bucket_contains(first, fingerprint) || self.bucket_contains(second, fingerprint) {
            return Ok(());
        }
        if self.bucket_insert(first, fingerprint) {
            return Ok(());
        }
        if self.bucket_insert(second, fingerprint) {
            return Ok(());
        }

        let mut bucket = first;
        let mut fp = self.bucket_swap_random_slot(bucket, fingerprint);
        bucket = self.alternate_bucket(bucket, fp);

        for _ in 0..self.kick.max_kicks {
            if self.bucket_insert(bucket, fp) {
                return Ok(());
            }
            let evicted = self.bucket_swap_random_slot(bucket, fp);
            fp = evicted;
            bucket = self.alternate_bucket(bucket, fp);
        }

        Err(SketchFull::new(fp))
    }

    /// Returns whether `item` may be present.
    pub fn contains_item<T>(&self, item: &T) -> bool
    where
        T: Hash + ?Sized,
    {
        let (fingerprint, first, second) = self.fingerprint_and_buckets(item);
        self.bucket_contains(first, fingerprint) || self.bucket_contains(second, fingerprint)
    }

    /// Removes `item`, returning whether a fingerprint was cleared. Only
    /// remove items known to be present (see the twin caveat on
    /// [`CuckooFilter`](crate::sketch::CuckooFilter)).
    pub fn remove_item<T>(&mut self, item: &T) -> bool
    where
        T: Hash + ?Sized,
    {
        let (fingerprint, first, second) = self.fingerprint_and_buckets(item);
        self.bucket_remove(first, fingerprint) || self.bucket_remove(second, fingerprint)
    }

    fn fingerprint_and_buckets<T>(&self, item: &T) -> (u64, usize, usize)
    where
        T: Hash + ?Sized,
    {
        let hash = hash_one(&self.hasher, item);
        let fingerprint =
            PartialKeyCuckoo::fingerprint(SEMI_SORTED_FINGERPRINT_BITS, hash);
        let first = PartialKeyCuckoo::bucket(hash, self.geometry.buckets);
        let second = self.alternate_bucket(first, fingerprint);
        (fingerprint, first, second)
    }
}

impl<S> Sketch for SemiSortedCuckooFilter<S> {
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

impl<T, S> Insert<T> for SemiSortedCuckooFilter<S>
where
    T: Hash + ?Sized,
    S: BuildHasher,
{
    type Err = SketchFull;

    fn insert(&mut self, item: &T) -> Result<(), Self::Err> {
        self.insert_item(item)
    }
}

impl<T, S> Contains<T> for SemiSortedCuckooFilter<S>
where
    T: Hash + ?Sized,
    S: BuildHasher,
{
    fn contains(&self, item: &T) -> bool {
        self.contains_item(item)
    }
}

impl<T, S> Remove<T> for SemiSortedCuckooFilter<S>
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
    use super::{SEMI_SORTED_FINGERPRINT_BITS, SemiSortedCuckooFilter, rank4, unrank4};
    use crate::policy::{RngLite, XorShift64};
    use crate::sketch::CuckooGeometry;
    use crate::traits::{Contains, Insert, Remove, Sketch};

    #[test]
    fn rank_unrank_roundtrips_random_multisets() {
        let mut rng = XorShift64::new(11);
        for _ in 0..20_000 {
            let mut slots = [
                rng.next_index(1 << SEMI_SORTED_FINGERPRINT_BITS) as u64,
                rng.next_index(1 << SEMI_SORTED_FINGERPRINT_BITS) as u64,
                rng.next_index(1 << SEMI_SORTED_FINGERPRINT_BITS) as u64,
                rng.next_index(1 << SEMI_SORTED_FINGERPRINT_BITS) as u64,
            ];
            slots.sort_unstable();
            assert_eq!(unrank4(rank4(slots)), slots, "roundtrip failed for {slots:?}");
        }
    }

    #[test]
    fn rank_unrank_edge_cases() {
        assert_eq!(rank4([0, 0, 0, 0]), 0);
        assert_eq!(unrank4(0), [0, 0, 0, 0]);
        let max = (1 << SEMI_SORTED_FINGERPRINT_BITS) - 1;
        assert_eq!(unrank4(rank4([max, max, max, max])), [max, max, max, max]);
        assert_eq!(unrank4(rank4([0, 1, 2, 3])), [0, 1, 2, 3]);
    }

    fn test_geometry() -> CuckooGeometry {
        CuckooGeometry {
            buckets: 64,
            slots_per_bucket: 4,
            fingerprint_bits: SEMI_SORTED_FINGERPRINT_BITS,
            max_kicks: 100,
        }
    }

    #[test]
    fn insert_contains_and_remove() {
        let mut filter = SemiSortedCuckooFilter::from_geometry(test_geometry(), 3);
        filter.insert_item(&42_u64).unwrap();

        assert!(filter.contains_item(&42_u64));
        assert_eq!(filter.occupancy(), 1);
        assert!(filter.remove_item(&42_u64));
        assert!(!filter.contains_item(&42_u64));
        assert_eq!(filter.occupancy(), 0);
    }

    #[test]
    fn no_false_negatives_at_high_load() {
        let mut filter = SemiSortedCuckooFilter::with_capacity(1_000, 0.001);
        for i in 0..900_u64 {
            filter.insert_item(&i).unwrap();
        }
        for i in 0..900_u64 {
            assert!(filter.contains_item(&i), "missing {i}");
        }
        for i in (0..900_u64).step_by(2) {
            assert!(filter.remove_item(&i), "removing {i}");
        }
        for i in (1..900_u64).step_by(2) {
            assert!(filter.contains_item(&i), "missing {i} after deletions");
        }
    }

    #[test]
    fn duplicate_inserts_are_deduplicated() {
        let mut filter = SemiSortedCuckooFilter::from_geometry(test_geometry(), 5);
        for _ in 0..10 {
            filter.insert_item(&7_u64).unwrap();
        }
        assert_eq!(filter.occupancy(), 1);
        assert!(filter.remove_item(&7_u64));
        assert_eq!(filter.occupancy(), 0);
    }

    #[test]
    fn tiny_filter_can_report_full() {
        let geometry = CuckooGeometry {
            buckets: 1,
            slots_per_bucket: 4,
            fingerprint_bits: SEMI_SORTED_FINGERPRINT_BITS,
            max_kicks: 2,
        };
        let mut filter = SemiSortedCuckooFilter::from_geometry(geometry, 0);
        let mut saw_full = false;
        for item in 0..100_u64 {
            if filter.insert_item(&item).is_err() {
                saw_full = true;
                break;
            }
        }
        assert!(saw_full);
    }

    #[test]
    fn storage_is_56_bits_per_bucket() {
        let filter = SemiSortedCuckooFilter::from_geometry(test_geometry(), 0);
        assert_eq!(filter.storage_bytes(), 64 * 7);
    }

    #[test]
    fn capability_traits_work() {
        let mut filter = SemiSortedCuckooFilter::with_capacity(128, 0.001);
        Insert::<str>::insert(&mut filter, "alice").unwrap();
        assert!(Contains::<str>::contains(&filter, "alice"));
        assert!(Remove::<str>::remove(&mut filter, "alice"));
        assert_eq!(Sketch::len_hint(&filter), Some(0));
    }

    #[test]
    #[should_panic(expected = "15-bit fingerprints")]
    fn geometry_requires_15_bit_fingerprints() {
        let geometry = CuckooGeometry {
            buckets: 64,
            slots_per_bucket: 4,
            fingerprint_bits: 16,
            max_kicks: 100,
        };
        SemiSortedCuckooFilter::from_geometry(geometry, 0);
    }
}
