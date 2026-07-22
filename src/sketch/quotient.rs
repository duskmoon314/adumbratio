//! Quotient filter implementation.

use core::hash::{BuildHasher, Hash};

use alloc::vec::Vec;

use crate::block::{BitArray, PackedArray};
use crate::error::{MergeError, SketchFull};
use crate::hash::{DefaultBuildHasher, hash_one};
use crate::traits::{Contains, Insert, Merge, Remove, Sketch};

/// Explicit quotient-filter geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct QuotientGeometry {
    /// Log2 of the slot count; the table holds `2^quotient_bits` slots.
    pub quotient_bits: u32,
}

/// A quotient filter for approximate membership with deletion and merging.
///
/// The item hash is split into a *quotient* (the home slot index) and a
/// small *remainder*. Collisions are resolved by Robin-Hood-style runs: a
/// run stores the remainders of one quotient contiguously and in sorted
/// order, starting at the home slot if possible, shifted right otherwise.
/// Three metadata bits per slot keep the structure decodable:
///
/// ```text
/// hash("x") = [ q quotient bits | R remainder bits | ... ]
///
/// slots:  [... | r2 | r1 | r3 | ...]
///                home of run 1, shifted right by one
///
/// occupied(j)     = some run's home is slot j
/// continuation(j) = slot j is not the first element of its run
/// shifted(j)      = slot j's element does not live at its home
/// ```
///
/// Compared with the cuckoo filter, the quotient filter is *mergeable* (by
/// re-inserting the decoded pairs) and cache-linear, at the cost of slower
/// inserts as clusters grow. Deletion removes one matching remainder from
/// the item's run; only delete items you inserted, as with cuckoo.
///
/// Clusters are decoded and re-encoded whole rather than shifted in place:
/// the first run of a cluster is always unshifted, so a cluster is exactly
/// a sorted list of `(quotient, remainder)` pairs — rewriting it is correct
/// by construction and cheap at the load factors this filter targets.
///
/// # References
///
/// - Michael A. Bender, Martin Farach-Colton, Rob Johnson, et al., "Don't
///   Thrash: How to Cache Your Hash on Flash", PVLDB 2012.
///   <https://doi.org/10.14778/2350229.2350275>
/// - Prashant Pandey, Michael A. Bender, Rob Johnson, and Rob Patro, "A
///   General-Purpose Counting Filter: Making Every Bit Count", SIGMOD 2017.
///   <https://doi.org/10.1145/3035918.3035963>
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct QuotientFilter<const R: u32 = 10, S = DefaultBuildHasher> {
    remainders: PackedArray<R>,
    occupied: BitArray,
    continuation: BitArray,
    shifted: BitArray,
    quotient_bits: u32,
    len: usize,
    seed_fingerprint: u64,
    hasher: S,
}

impl QuotientFilter<10, DefaultBuildHasher> {
    /// Creates a quotient filter sized for an expected item count and
    /// target FPP, with seed zero.
    ///
    /// The table targets a 90% load factor; the FPP is about `1/2^R`.
    ///
    /// # Panics
    ///
    /// Panics if `expected_items` is zero or if `false_positive_rate` is not
    /// in `0.0..1.0` or cannot be met with 10-bit remainders.
    pub fn with_capacity(expected_items: u64, false_positive_rate: f64) -> Self {
        Self::with_capacity_and_seed(expected_items, false_positive_rate, 0)
    }

    /// Creates a seeded quotient filter from capacity and target FPP.
    ///
    /// # Panics
    ///
    /// Panics under the same conditions as [`Self::with_capacity`].
    pub fn with_capacity_and_seed(
        expected_items: u64,
        false_positive_rate: f64,
        seed: u64,
    ) -> Self {
        Self::from_geometry(
            QuotientGeometry::for_capacity(expected_items, false_positive_rate),
            seed,
        )
    }
}

impl<const R: u32> QuotientFilter<R, DefaultBuildHasher> {
    /// Creates a quotient filter with explicit geometry and seed.
    ///
    /// # Panics
    ///
    /// Panics if the geometry is invalid or incompatible with `R`.
    pub fn from_geometry(geometry: QuotientGeometry, seed: u64) -> Self {
        let hasher = DefaultBuildHasher::new(seed);
        Self::from_parts(geometry, hasher.seed_fingerprint(), hasher)
    }
}

impl<const R: u32, S> QuotientFilter<R, S> {
    /// Creates a quotient filter from explicit components.
    ///
    /// # Panics
    ///
    /// Panics if `quotient_bits` is zero or above 32, or if
    /// `quotient_bits + R > 64`.
    pub fn from_parts(geometry: QuotientGeometry, seed_fingerprint: u64, hasher: S) -> Self {
        geometry.validate::<R>();
        let slots = 1 << geometry.quotient_bits;
        Self {
            remainders: PackedArray::new(slots),
            occupied: BitArray::new(slots),
            continuation: BitArray::new(slots),
            shifted: BitArray::new(slots),
            quotient_bits: geometry.quotient_bits,
            len: 0,
            seed_fingerprint,
            hasher,
        }
    }

    /// Returns the realized geometry.
    pub const fn geometry(&self) -> QuotientGeometry {
        QuotientGeometry {
            quotient_bits: self.quotient_bits,
        }
    }

    /// Returns the seed fingerprint used by merge compatibility checks.
    pub const fn seed_fingerprint(&self) -> u64 {
        self.seed_fingerprint
    }

    /// Returns the number of stored items.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the filter is empty.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of table slots.
    pub const fn table_len(&self) -> usize {
        1 << self.quotient_bits
    }

    /// Returns the occupied fraction of slots.
    pub fn load_factor(&self) -> f64 {
        self.len as f64 / self.table_len() as f64
    }

    /// Returns the approximate false-positive probability, about `1/2^R` at
    /// the load factors this filter targets.
    pub fn expected_fpp(&self) -> f64 {
        (1.0 / (1_u64 << R.min(63)) as f64).min(1.0)
    }

    /// Returns the byte length of the table storage.
    pub fn storage_bytes(&self) -> usize {
        self.remainders.storage_bytes()
            + self.occupied.storage_bytes()
            + self.continuation.storage_bytes()
            + self.shifted.storage_bytes()
    }

    /// Clears the table.
    pub fn clear(&mut self) {
        self.remainders.clear();
        self.occupied.clear();
        self.continuation.clear();
        self.shifted.clear();
        self.len = 0;
    }

    /// Splits a hash into home slot and remainder.
    fn split(&self, hash: u64) -> (usize, u64) {
        let quotient = (hash >> (64 - self.quotient_bits)) as usize & (self.table_len() - 1);
        let remainder = (hash >> (64 - self.quotient_bits - R)) & PackedArray::<R>::MAX;
        (quotient, remainder)
    }

    /// A slot is empty when all three metadata bits are clear; a stored
    /// remainder of zero always carries at least one set bit.
    fn slot_empty(&self, slot: usize) -> bool {
        !self.occupied.get(slot) && !self.continuation.get(slot) && !self.shifted.get(slot)
    }

    /// Returns the first slot of the cluster containing `slot`. On a
    /// completely full table every slot is one cluster; then the first
    /// run-start slot is returned so decoding stays aligned.
    fn cluster_start(&self, slot: usize) -> usize {
        let mask = self.table_len() - 1;
        let mut start = slot;
        for _ in 0..self.table_len() {
            if self.slot_empty((start + mask) & mask) {
                return start;
            }
            start = (start + mask) & mask;
        }
        // Full table: advance to an unshifted run start (a slot holding a
        // run's first element at its own home), so decoding stays aligned
        // with the occupied-bit order. Bounded to one wrap.
        let mut scanned = 0_usize;
        while (self.continuation.get(start) || self.shifted.get(start))
            && scanned < self.table_len()
        {
            start = (start + 1) & mask;
            scanned += 1;
        }
        start
    }

    /// Decodes the cluster starting at `start` into its sorted
    /// `(quotient, remainder)` pairs. Runs and occupied bits appear in the
    /// same order, so the i-th run's home is the i-th occupied slot. Both
    /// scans are bounded to one full wrap for the full-table case.
    fn decode_cluster(&self, start: usize) -> Vec<(u64, u64)> {
        let mask = self.table_len() - 1;
        let mut pairs = Vec::new();
        let mut homes = Vec::new();
        let mut slot = start;
        for _ in 0..self.table_len() {
            if self.slot_empty(slot) {
                break;
            }
            if self.occupied.get(slot) {
                homes.push(slot as u64);
            }
            slot = (slot + 1) & mask;
        }
        let mut run = 0_usize;
        let mut slot = start;
        for _ in 0..self.table_len() {
            if self.slot_empty(slot) {
                break;
            }
            if !self.continuation.get(slot) && slot != start {
                run += 1;
            }
            pairs.push((homes[run], self.remainders.get(slot)));
            slot = (slot + 1) & mask;
        }
        pairs
    }

    /// Sorts pairs in cluster order: by circular distance of the quotient
    /// from the cluster start (wrapping around the table end), so encoding
    /// preserves the run layout instead of re-anchoring at the numerically
    /// smallest home.
    fn circular_sort(&self, start: usize, pairs: &mut [(u64, u64)]) {
        let mask = self.table_len() - 1;
        pairs.sort_unstable_by_key(|&(quotient, remainder)| {
            (
                (quotient as usize + self.table_len() - start) & mask,
                remainder,
            )
        });
    }

    /// Re-encodes `pairs` (in cluster order, see [`Self::circular_sort`]),
    /// replacing the `old_len`-slot cluster at `start`. Clearing covers
    /// both the old region and the new one: the first run's home may sit
    /// before `start` (full-table wrap) or after it (deleted first run).
    /// Each run starts at its home if free, otherwise directly after the
    /// previous run, so every element sits at or circularly after its home,
    /// and the region may split into several clusters when gaps reappear.
    ///
    /// Debug builds self-check every encode by re-decoding each written
    /// pair through the query path.
    fn encode_cluster(&mut self, start: usize, old_len: usize, pairs: &[(u64, u64)]) {
        let mask = self.table_len() - 1;
        let mut clear_len = old_len;
        if let Some(&(first_home, _)) = pairs.first() {
            let home_offset = (first_home as usize + self.table_len() - start) & mask;
            clear_len = clear_len.max(home_offset + pairs.len());
        }
        for i in 0..clear_len {
            let slot = (start + i) & mask;
            self.remainders.set(slot, 0);
            self.occupied.unset(slot);
            self.continuation.unset(slot);
            self.shifted.unset(slot);
        }
        let Some(&(first_home, _)) = pairs.first() else {
            return;
        };
        // Offsets are circular distances from the first run's home.
        let mut pos = 0_usize;
        let mut previous_quotient = None;
        for &(quotient, remainder) in pairs {
            let home_offset = (quotient as usize + self.table_len() - first_home as usize) & mask;
            if previous_quotient != Some(quotient) && pos < home_offset {
                pos = home_offset; // jump to the home slot, leaving a gap
            }
            let slot = (first_home as usize + pos) & mask;
            self.remainders.set(slot, remainder);
            if previous_quotient == Some(quotient) {
                self.continuation.set(slot);
            }
            if slot != quotient as usize {
                self.shifted.set(slot);
            }
            self.occupied.set(quotient as usize);
            previous_quotient = Some(quotient);
            pos += 1;
        }

        #[cfg(debug_assertions)]
        {
            // Semantic self-check: every pair just encoded must be findable
            // through the same decode path queries use.
            for &(quotient, remainder) in pairs {
                let start = self.cluster_start(quotient as usize);
                let decoded = self.decode_cluster(start);
                let found = decoded
                    .iter()
                    .any(|&(q, r)| q == quotient && r == remainder);
                debug_assert!(
                    found,
                    "encoded pair ({quotient}, {remainder}) not findable: cluster_start {start}, decoded {decoded:?}"
                );
            }
        }
    }
}

impl<const R: u32, S> QuotientFilter<R, S>
where
    S: BuildHasher,
{
    /// Inserts `item`.
    ///
    /// If the item's `(quotient, remainder)` pair is already stored, the
    /// insert is deduplicated and succeeds immediately.
    ///
    /// # Errors
    ///
    /// Returns [`SketchFull`] when the table has no empty slot left.
    pub fn insert_item<T>(&mut self, item: &T) -> Result<(), SketchFull>
    where
        T: Hash + ?Sized,
    {
        if self.len == self.table_len() {
            return Err(SketchFull::new(hash_one(&self.hasher, item)));
        }
        let (quotient, remainder) = self.split(hash_one(&self.hasher, item));
        if self.slot_empty(quotient) {
            self.remainders.set(quotient, remainder);
            self.occupied.set(quotient);
            self.len += 1;
            return Ok(());
        }

        let start = self.cluster_start(quotient);
        let mut pairs = self.decode_cluster(start);
        if pairs.contains(&(quotient as u64, remainder)) {
            return Ok(());
        }
        pairs.push((quotient as u64, remainder));
        self.circular_sort(start, &mut pairs);
        let old_len = pairs.len() - 1;
        self.encode_cluster(start, old_len, &pairs);
        self.len += 1;
        Ok(())
    }

    /// Returns whether `item` may be present.
    pub fn contains_item<T>(&self, item: &T) -> bool
    where
        T: Hash + ?Sized,
    {
        let (quotient, remainder) = self.split(hash_one(&self.hasher, item));
        if !self.occupied.get(quotient) {
            return false;
        }
        let start = self.cluster_start(quotient);
        let pairs = self.decode_cluster(start);
        pairs
            .iter()
            .any(|&(q, r)| q == quotient as u64 && r == remainder)
    }

    /// Removes one stored occurrence of `item`, returning whether a
    /// `(quotient, remainder)` pair was cleared.
    ///
    /// Removing an item that was never inserted may clear a pair belonging
    /// to a different item; only remove items known to be present.
    pub fn remove_item<T>(&mut self, item: &T) -> bool
    where
        T: Hash + ?Sized,
    {
        let (quotient, remainder) = self.split(hash_one(&self.hasher, item));
        if !self.occupied.get(quotient) {
            return false;
        }
        let start = self.cluster_start(quotient);
        let mut pairs = self.decode_cluster(start);
        let Some(position) = pairs
            .iter()
            .position(|&(q, r)| q == quotient as u64 && r == remainder)
        else {
            return false;
        };
        pairs.remove(position);
        // Decode yields cluster order already; removal preserves it.
        let old_len = pairs.len() + 1;
        self.encode_cluster(start, old_len, &pairs);
        self.len -= 1;
        true
    }

    /// Returns every stored `(quotient, remainder)` pair, used by merge.
    fn all_pairs(&self) -> Vec<(u64, u64)> {
        let mut pairs = Vec::with_capacity(self.len);
        let mut slot = 0;
        while slot < self.table_len() {
            if self.slot_empty(slot) {
                slot += 1;
                continue;
            }
            let cluster = self.decode_cluster(slot);
            slot += cluster.len();
            pairs.extend(cluster);
        }
        pairs
    }
}

impl QuotientGeometry {
    /// Computes quotient-filter geometry from capacity and target FPP,
    /// targeting a 90% load factor.
    ///
    /// # Panics
    ///
    /// Panics if `expected_items` is zero or if `false_positive_rate` is not
    /// in `0.0..1.0`.
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
        // Slots = ceil(n / 0.9) in exact integer arithmetic (no float math).
        let slots = (10 * expected_items as u128).div_ceil(9);
        assert!(
            slots <= (1_u128 << 32),
            "computed quotient-filter slot count exceeds 2^32"
        );
        let slots = (slots.max(1) as usize).next_power_of_two();
        Self {
            quotient_bits: slots.trailing_zeros(),
        }
    }

    /// Validates the geometry against the remainder width `R`.
    ///
    /// # Panics
    ///
    /// Panics if `quotient_bits` is zero or above 32, or if
    /// `quotient_bits + R > 64`.
    pub fn validate<const R: u32>(self) {
        assert!(
            (1..=32).contains(&self.quotient_bits),
            "quotient bits must be in 1..=32"
        );
        assert!(
            self.quotient_bits + R <= 64,
            "quotient bits + remainder width must be at most 64"
        );
    }
}

impl<const R: u32, S> Sketch for QuotientFilter<R, S> {
    fn clear(&mut self) {
        self.clear();
    }

    fn len_hint(&self) -> Option<u64> {
        Some(self.len as u64)
    }

    fn storage_bytes(&self) -> usize {
        self.storage_bytes()
    }
}

impl<T, const R: u32, S> Insert<T> for QuotientFilter<R, S>
where
    T: Hash + ?Sized,
    S: BuildHasher,
{
    type Err = SketchFull;

    fn insert(&mut self, item: &T) -> Result<(), Self::Err> {
        self.insert_item(item)
    }
}

impl<T, const R: u32, S> Contains<T> for QuotientFilter<R, S>
where
    T: Hash + ?Sized,
    S: BuildHasher,
{
    fn contains(&self, item: &T) -> bool {
        self.contains_item(item)
    }
}

impl<T, const R: u32, S> Remove<T> for QuotientFilter<R, S>
where
    T: Hash + ?Sized,
    S: BuildHasher,
{
    fn remove(&mut self, item: &T) -> bool {
        self.remove_item(item)
    }
}

impl<const R: u32, S> Merge for QuotientFilter<R, S>
where
    S: BuildHasher,
{
    /// Merges by decoding every pair of `other` and inserting it. Run-level
    /// merging without decoding is a possible future optimization.
    fn merge_from(&mut self, other: &Self) -> Result<(), MergeError> {
        if self.geometry() != other.geometry() {
            return Err(MergeError::GeometryMismatch);
        }
        if self.seed_fingerprint != other.seed_fingerprint {
            return Err(MergeError::SeedMismatch);
        }
        for (quotient, remainder) in other.all_pairs() {
            if self.len == self.table_len() {
                return Err(MergeError::InsufficientCapacity);
            }
            if self.slot_empty(quotient as usize) {
                self.remainders.set(quotient as usize, remainder);
                self.occupied.set(quotient as usize);
                self.len += 1;
                continue;
            }
            let start = self.cluster_start(quotient as usize);
            let mut pairs = self.decode_cluster(start);
            if pairs.contains(&(quotient, remainder)) {
                continue;
            }
            pairs.push((quotient, remainder));
            self.circular_sort(start, &mut pairs);
            let old_len = pairs.len() - 1;
            self.encode_cluster(start, old_len, &pairs);
            self.len += 1;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{QuotientFilter, QuotientGeometry};
    use crate::error::MergeError;
    use crate::traits::{Contains, Insert, Merge, Remove, Sketch};

    #[test]
    fn insert_contains_remove_roundtrip() {
        let mut filter = QuotientFilter::with_capacity(1_000, 0.001);
        filter.insert_item(&42_u64).unwrap();

        assert!(filter.contains_item(&42_u64));
        assert!(filter.remove_item(&42_u64));
        assert!(!filter.contains_item(&42_u64));
        assert!(!filter.remove_item(&42_u64));
    }

    #[test]
    fn no_false_negatives_with_heavy_collisions() {
        // A tiny table forces long clusters, exercising decode/encode.
        let mut filter = QuotientFilter::<10>::from_geometry(QuotientGeometry { quotient_bits: 4 }, 3);
        let n = 12_u64; // 75% load of 16 slots
        for i in 0..n {
            filter.insert_item(&i).unwrap();
        }
        for i in 0..n {
            assert!(filter.contains_item(&i), "missing {i}");
        }
        for i in (0..n).step_by(2) {
            assert!(filter.remove_item(&i), "removing {i}");
        }
        for i in (1..n).step_by(2) {
            assert!(filter.contains_item(&i), "missing {i} after deletions");
        }
    }

    #[test]
    fn duplicate_inserts_are_deduplicated() {
        let mut filter = QuotientFilter::with_capacity(100, 0.001);
        for _ in 0..10 {
            filter.insert_item(&7_u64).unwrap();
        }
        assert_eq!(filter.len(), 1);
        assert!(filter.remove_item(&7_u64));
        assert_eq!(filter.len(), 0);
    }

    #[test]
    fn full_table_reports_full() {
        let mut filter = QuotientFilter::<10>::from_geometry(QuotientGeometry { quotient_bits: 2 }, 0);
        let mut saw_full = false;
        for i in 0..100_u64 {
            if filter.insert_item(&i).is_err() {
                saw_full = true;
                break;
            }
        }
        assert!(saw_full);
    }

    #[test]
    fn merge_combines_filters() {
        let geometry = QuotientGeometry { quotient_bits: 8 };
        let mut left = QuotientFilter::<10>::from_geometry(geometry, 5);
        let mut right = QuotientFilter::<10>::from_geometry(geometry, 5);
        for i in 0..100_u64 {
            left.insert_item(&i).unwrap();
        }
        for i in 100..200_u64 {
            right.insert_item(&i).unwrap();
        }

        left.merge_from(&right).unwrap();
        assert_eq!(left.len(), 200);
        for i in 0..200_u64 {
            assert!(left.contains_item(&i));
        }

        let other_seed = QuotientFilter::<10>::from_geometry(geometry, 6);
        assert_eq!(left.merge_from(&other_seed), Err(MergeError::SeedMismatch));
    }

    #[test]
    fn capability_traits_work() {
        let mut filter = QuotientFilter::with_capacity(100, 0.001);
        Insert::<str>::insert(&mut filter, "alice").unwrap();
        assert!(Contains::<str>::contains(&filter, "alice"));
        assert!(Remove::<str>::remove(&mut filter, "alice"));
        assert_eq!(Sketch::len_hint(&filter), Some(0));
    }
}
