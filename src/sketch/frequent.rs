//! Deterministic frequent-element sketches: Misra–Gries and Space-Saving.

use core::convert::Infallible;
use core::hash::Hash;

use alloc::vec::Vec;

use crate::error::MergeError;
use crate::traits::{Insert, Merge, Sketch};

/// A Misra–Gries summary: `k` counters maintained by the decrement-all
/// rule, for deterministic heavy hitters.
///
/// Inserting an item increments its counter if tracked, starts a counter
/// if one is free, and otherwise decrements *every* counter. The summary
/// never overestimates, and its guarantee is deterministic (no
/// probabilities, no hashes):
///
/// ```text
/// insert("x"): tracked? +1 : free slot? start at 1 : all counters -1
///
/// estimate("x") in [f(x) - N/(k+1), f(x)]   for tracked x
/// every item with f(x) > N/(k+1) is tracked
/// ```
///
/// Counters live in a small `Vec`, so one insert costs `O(k)` comparisons
/// — the classic trade for a hash-free, seed-free summary. Merging sums
/// counters pairwise and then debits every counter by the `(k+1)`-th
/// largest summed count, the standard Misra–Gries merge.
///
/// # References
///
/// - Jayadev Misra and David Gries, "Finding repeated elements", Science
///   of Computer Programming 2(2), 1982. <https://doi.org/10.1016/0167-6423(82)90012-0>
/// - Richard M. Karp, Scott Shenker, and Christos H. Papadimitriou, "A
///   Simple Algorithm for Finding Frequent Elements in Streams and Bags",
///   ACM TODS 28(1), 2003. <https://doi.org/10.1145/762471.762473>
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MisraGries<T> {
    counters: Vec<(T, u64)>,
    k: usize,
    total: u64,
}

/// A Space-Saving summary: `k` `(item, count, error)` triples maintained
/// by the replace-the-minimum rule.
///
/// Inserting an item increments its counter if tracked, starts one if a
/// slot is free, and otherwise *evicts the minimum* entry: the new item
/// inherits `count = min + 1` and `error = min`, so overestimation is
/// bounded by the recorded error. The guarantees are deterministic:
///
/// ```text
/// insert("x"): tracked? +1 : free? start at 1 : evict min, keep min+1 (error min)
///
/// f(x) <= estimate(x) <= f(x) + error(x),   error <= N/(k+1)
/// every item with f(x) > N/(k+1) is tracked
/// ```
///
/// Space-Saving never underestimates — the dual of Misra–Gries — and in
/// practice tracks heavy hitters more accurately on skewed streams.
/// Merging replays the other summary's entries as weighted inserts, the
/// construction the paper analyzes.
///
/// # References
///
/// - Ahmed Metwally, Divyakant Agrawal, and Amr El Abbadi, "Efficient
///   Computation of Frequent and Top-k Elements in Data Streams", ICDT
///   2005. <https://doi.org/10.1007/978-3-540-30570-5_27>
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpaceSaving<T> {
    entries: Vec<(T, u64, u64)>,
    k: usize,
    total: u64,
}

impl<T> MisraGries<T> {
    /// Creates a Misra–Gries summary with `k` counters.
    ///
    /// # Panics
    ///
    /// Panics if `k` is zero.
    pub fn new(k: usize) -> Self {
        assert!(k > 0, "Misra-Gries needs at least one counter");
        Self {
            counters: Vec::new(),
            k,
            total: 0,
        }
    }

    /// Returns the number of counters.
    pub const fn k(&self) -> usize {
        self.k
    }

    /// Returns the total number of inserted events.
    pub const fn total_count(&self) -> u64 {
        self.total
    }

    /// Returns the deterministic error bound, `N / (k + 1)`.
    pub const fn error_bound(&self) -> u64 {
        self.total / (self.k as u64 + 1)
    }

    /// Returns the tracked counters, in no particular order.
    pub fn counters(&self) -> &[(T, u64)] {
        &self.counters
    }

    /// Returns the tracked counters with their estimates, heaviest first.
    pub fn top_k(&self) -> Vec<(T, u64)>
    where
        T: Clone,
    {
        let mut items = self.counters.clone();
        items.sort_by_key(|item| core::cmp::Reverse(item.1));
        items
    }

    /// Clears the summary.
    pub fn clear(&mut self) {
        self.counters.clear();
        self.total = 0;
    }

    /// Returns the byte length of the counter storage.
    pub fn storage_bytes(&self) -> usize {
        self.counters.len() * (size_of::<T>() + size_of::<u64>())
    }

    /// Debits every counter by `debit`, dropping non-positive ones (the
    /// merge step).
    fn debit_all(&mut self, debit: u64) {
        if debit == 0 {
            return;
        }
        self.counters
            .retain_mut(|entry| match entry.1.checked_sub(debit) {
                Some(remaining) if remaining > 0 => {
                    entry.1 = remaining;
                    true
                }
                _ => false,
            });
    }
}

impl<T: Eq> MisraGries<T> {
    /// Inserts `count` occurrences of `item` at once, as one event of
    /// weight `count`.
    ///
    /// Note this is *not* equivalent to calling [`Self::insert_item`]
    /// `count` times: when the item is untracked and the summary is full,
    /// the weighted form debits every counter by the whole `count` in one
    /// step, while iterated inserts would debit progressively with
    /// different intermediate counters. For tracked items (the common
    /// case) the two coincide.
    pub fn insert_count(&mut self, item: &T, count: u64)
    where
        T: Clone,
    {
        if count == 0 {
            return;
        }
        self.total += count;
        if let Some(entry) = self.counters.iter_mut().find(|entry| entry.0 == *item) {
            entry.1 += count;
            return;
        }
        if self.counters.len() < self.k {
            self.counters.push((item.clone(), count));
            return;
        }
        // Weighted form of the decrement-all rule: the item's weight is
        // consumed by debiting every counter.
        self.debit_all(count);
    }

    /// Inserts one occurrence of `item`.
    pub fn insert_item(&mut self, item: &T)
    where
        T: Clone,
    {
        self.total += 1;
        if let Some(entry) = self.counters.iter_mut().find(|entry| entry.0 == *item) {
            entry.1 += 1;
            return;
        }
        if self.counters.len() < self.k {
            self.counters.push((item.clone(), 1));
            return;
        }
        self.debit_all(1);
    }

    /// Returns the estimate for `item`: its counter if tracked, else zero.
    /// The true frequency lies in `[estimate, estimate + N/(k+1)]`... more
    /// precisely in `[f - N/(k+1), f]` for tracked items.
    pub fn estimate_item(&self, item: &T) -> u64 {
        self.counters
            .iter()
            .find(|entry| entry.0 == *item)
            .map(|entry| entry.1)
            .unwrap_or(0)
    }
}

impl<T> SpaceSaving<T> {
    /// Creates a Space-Saving summary with `k` entries.
    ///
    /// # Panics
    ///
    /// Panics if `k` is zero.
    pub fn new(k: usize) -> Self {
        assert!(k > 0, "Space-Saving needs at least one entry");
        Self {
            entries: Vec::new(),
            k,
            total: 0,
        }
    }

    /// Returns the number of entries.
    pub const fn k(&self) -> usize {
        self.k
    }

    /// Returns the total number of inserted events.
    pub const fn total_count(&self) -> u64 {
        self.total
    }

    /// Returns the deterministic error bound, `N / (k + 1)`.
    pub const fn error_bound(&self) -> u64 {
        self.total / (self.k as u64 + 1)
    }

    /// Returns the tracked `(item, count, error)` entries, heaviest first.
    pub fn top_k(&self) -> Vec<(T, u64)>
    where
        T: Clone,
    {
        let mut items: Vec<_> = self
            .entries
            .iter()
            .map(|(item, count, _)| (item.clone(), *count))
            .collect();
        items.sort_by_key(|item| core::cmp::Reverse(item.1));
        items
    }

    /// Clears the summary.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.total = 0;
    }

    /// Returns the byte length of the entry storage.
    pub fn storage_bytes(&self) -> usize {
        self.entries.len() * (size_of::<T>() + 2 * size_of::<u64>())
    }

    /// Returns the index of a minimum-count entry.
    fn min_index(&self) -> usize {
        self.entries
            .iter()
            .enumerate()
            .min_by_key(|&(_, entry)| entry.1)
            .map(|(i, _)| i)
            .expect("entries are full and non-empty")
    }
}

impl<T: Eq> SpaceSaving<T> {
    /// Inserts `count` occurrences of `item` at once, as one event of
    /// weight `count` — the weighted form the paper's merge builds on.
    ///
    /// Note this is *not* equivalent to calling [`Self::insert_item`]
    /// `count` times: when the item is untracked and the summary is full,
    /// the weighted form replaces the minimum entry once with
    /// `min + count` (error `min`), while iterated inserts would churn
    /// through replacement chains with different intermediate minima.
    /// For tracked items (the common case) the two coincide.
    pub fn insert_count(&mut self, item: &T, count: u64)
    where
        T: Clone,
    {
        if count == 0 {
            return;
        }
        self.total += count;
        if let Some(entry) = self.entries.iter_mut().find(|entry| entry.0 == *item) {
            entry.1 += count;
            return;
        }
        if self.entries.len() < self.k {
            self.entries.push((item.clone(), count, 0));
            return;
        }
        let min = self.min_index();
        let (_, min_count, _) = self.entries[min];
        self.entries[min] = (item.clone(), min_count + count, min_count);
    }

    /// Inserts one occurrence of `item`.
    pub fn insert_item(&mut self, item: &T)
    where
        T: Clone,
    {
        self.insert_count(item, 1)
    }

    /// Returns the estimate for `item`: its count if tracked, else zero.
    /// The true frequency lies in `[estimate - error, estimate]`; see
    /// [`Self::estimate_with_error`].
    pub fn estimate_item(&self, item: &T) -> u64 {
        self.estimate_with_error(item).0
    }

    /// Returns the estimate for `item` with its recorded error: the true
    /// frequency of a tracked item lies in `[count - error, count]`.
    pub fn estimate_with_error(&self, item: &T) -> (u64, u64) {
        self.entries
            .iter()
            .find(|entry| entry.0 == *item)
            .map(|entry| (entry.1, entry.2))
            .unwrap_or((0, 0))
    }
}

impl<T> Sketch for MisraGries<T> {
    fn clear(&mut self) {
        self.clear();
    }

    fn len_hint(&self) -> Option<u64> {
        Some(self.total)
    }

    fn storage_bytes(&self) -> usize {
        self.storage_bytes()
    }
}

impl<T> Sketch for SpaceSaving<T> {
    fn clear(&mut self) {
        self.clear();
    }

    fn len_hint(&self) -> Option<u64> {
        Some(self.total)
    }

    fn storage_bytes(&self) -> usize {
        self.storage_bytes()
    }
}

impl<T> Insert<T> for MisraGries<T>
where
    T: Hash + Eq + Clone,
{
    type Err = Infallible;

    fn insert(&mut self, item: &T) -> Result<(), Self::Err> {
        self.insert_item(item);
        Ok(())
    }
}

impl<T> Insert<T> for SpaceSaving<T>
where
    T: Hash + Eq + Clone,
{
    type Err = Infallible;

    fn insert(&mut self, item: &T) -> Result<(), Self::Err> {
        self.insert_item(item);
        Ok(())
    }
}

impl<T: Eq + Clone> Merge for MisraGries<T> {
    /// Merges by summing counters pairwise and debiting all counters by the
    /// `(k+1)`-th largest summed count, the standard Misra–Gries merge that
    /// preserves the deterministic error bound.
    fn merge_from(&mut self, other: &Self) -> Result<(), MergeError> {
        if self.k != other.k {
            return Err(MergeError::GeometryMismatch);
        }
        self.total += other.total;
        for (item, count) in &other.counters {
            if let Some(entry) = self.counters.iter_mut().find(|entry| entry.0 == *item) {
                entry.1 += count;
            } else {
                self.counters.push((item.clone(), *count));
            }
        }
        if self.counters.len() > self.k {
            let mut counts: Vec<u64> = self.counters.iter().map(|entry| entry.1).collect();
            counts.sort_unstable_by_key(|&count| core::cmp::Reverse(count));
            let debit = counts[self.k]; // (k+1)-th largest, zero-indexed
            self.debit_all(debit);
        }
        Ok(())
    }
}

impl<T: Eq + Clone> Merge for SpaceSaving<T> {
    /// Merges by replaying the other summary's entries as weighted inserts,
    /// the construction the paper analyzes for concatenated streams.
    fn merge_from(&mut self, other: &Self) -> Result<(), MergeError> {
        if self.k != other.k {
            return Err(MergeError::GeometryMismatch);
        }
        let other_entries = other.entries.clone();
        for (item, count, _) in &other_entries {
            self.insert_count(item, *count);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::{MisraGries, SpaceSaving};
    use crate::error::MergeError;
    use crate::traits::{Insert, Merge, Sketch};

    fn skewed_stream() -> Vec<u64> {
        let mut stream = Vec::new();
        for item in 0..100_u64 {
            for _ in 0..(1_000 / (item + 1)) {
                stream.push(item);
            }
        }
        stream
    }

    #[test]
    fn misra_gries_tracks_every_item_above_the_bound() {
        let mut mg = MisraGries::new(10);
        for item in skewed_stream() {
            mg.insert_item(&item);
        }
        let bound = mg.error_bound();
        let truth = |item: u64| 1_000 / (item + 1);
        // Every item above the bound must be tracked, and estimates satisfy
        // f - bound <= estimate <= f.
        for item in 0..100_u64 {
            if truth(item) > bound {
                assert!(mg.estimate_item(&item) > 0, "item {item} above bound not tracked");
            }
            let estimate = mg.estimate_item(&item);
            if estimate > 0 {
                assert!(estimate <= truth(item));
                assert!(estimate + bound >= truth(item));
            }
        }
    }

    #[test]
    fn space_saving_estimates_within_recorded_error() {
        let mut ss = SpaceSaving::new(10);
        for item in skewed_stream() {
            ss.insert_item(&item);
        }
        let bound = ss.error_bound();
        let truth = |item: u64| 1_000 / (item + 1);
        for item in 0..100_u64 {
            if truth(item) > bound {
                assert!(ss.estimate_item(&item) > 0, "item {item} above bound not tracked");
            }
            let (count, error) = ss.estimate_with_error(&item);
            if count > 0 {
                assert!(count >= truth(item), "item {item}: count {count} < truth {}", truth(item));
                assert!(count <= truth(item) + error);
                assert!(error <= bound);
            }
        }
    }

    #[test]
    fn merges_preserve_the_error_bounds() {
        let stream = skewed_stream();
        let mut mg_left = MisraGries::new(10);
        let mut mg_right = MisraGries::new(10);
        let mut ss_left = SpaceSaving::new(10);
        let mut ss_right = SpaceSaving::new(10);
        for (i, &item) in stream.iter().enumerate() {
            if i % 2 == 0 {
                mg_left.insert_item(&item);
                ss_left.insert_item(&item);
            } else {
                mg_right.insert_item(&item);
                ss_right.insert_item(&item);
            }
        }
        let n = (mg_left.total_count() + mg_right.total_count()) as usize;
        mg_left.merge_from(&mg_right).unwrap();
        ss_left.merge_from(&ss_right).unwrap();
        assert_eq!(mg_left.total_count() as usize, n);
        assert_eq!(ss_left.total_count() as usize, n);
        // Merged summaries must still contain the heavy hitters.
        assert!(mg_left.estimate_item(&0_u64) > 0);
        assert!(ss_left.estimate_item(&0_u64) > 0);

        let other_k = MisraGries::<u64>::new(5);
        assert_eq!(mg_left.merge_from(&other_k), Err(MergeError::GeometryMismatch));
        let other_k = SpaceSaving::<u64>::new(5);
        assert_eq!(ss_left.merge_from(&other_k), Err(MergeError::GeometryMismatch));
    }

    #[test]
    fn weighted_insert_differs_from_iteration_on_the_miss_path() {
        // Tracked path: weighted and iterated inserts coincide exactly.
        let mut weighted = MisraGries::new(2);
        let mut iterated = MisraGries::new(2);
        for item in ["a", "b"] {
            weighted.insert_item(&item);
            iterated.insert_item(&item);
        }
        weighted.insert_count(&"a", 5);
        for _ in 0..5 {
            iterated.insert_item(&"a");
        }
        assert_eq!(weighted.counters(), iterated.counters());

        // Miss path on fresh summaries: a weight-3 event debits every
        // counter by 3 in one step (consuming the weight), while three unit
        // inserts drop the old items once and then start tracking the new
        // one.
        let mut weighted = MisraGries::new(2);
        let mut iterated = MisraGries::new(2);
        for item in ["a", "b"] {
            weighted.insert_item(&item);
            iterated.insert_item(&item);
        }
        weighted.insert_count(&"c", 3);
        for _ in 0..3 {
            iterated.insert_item(&"c");
        }
        assert!(
            !weighted.counters().iter().any(|(item, _)| *item == "c"),
            "the weight-3 event must be consumed by the debits"
        );
        assert_eq!(iterated.estimate_item(&"c"), 2);
    }

    #[test]
    fn capability_traits_work() {
        let mut mg = MisraGries::<u64>::new(3);
        Insert::<u64>::insert(&mut mg, &7).unwrap();
        assert!(mg.estimate_item(&7) >= 1);
        assert_eq!(Sketch::len_hint(&mg), Some(1));

        let mut ss = SpaceSaving::<u64>::new(3);
        Insert::<u64>::insert(&mut ss, &7).unwrap();
        assert!(ss.estimate_item(&7) >= 1);
    }
}
