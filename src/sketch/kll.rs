//! KLL quantile sketch implementation.

use core::convert::Infallible;

use alloc::vec;
use alloc::vec::Vec;

use crate::error::MergeError;
use crate::policy::{RngLite, XorShift64};
use crate::traits::{Insert, Merge, Sketch};

/// A KLL sketch for approximate quantile and rank queries.
///
/// KLL answers "which value sits at rank `q` of the stream?" (median,
/// p95, …) and the inverse "what is the normalized rank of `x`?" with rank
/// error around `1/k`, storing only about `k + log2(n)` values instead of
/// the whole stream. The structure is a hierarchy of compactors: level `h`
/// holds up to `k` values of weight `2^h`, and when it overflows, its
/// contents are sorted and every other value (a randomly chosen parity) is
/// promoted to the next level.
///
/// ```text
/// insert(x) -> level 0 buffer
///                  | overflow
///                  v
///   sort, keep parity p in {0,1} at random, promote to level 1
///                  | overflow
///                  v
///   same compaction, weight doubles per level
///
/// quantile(q) = the value whose cumulative weight covers q * (n - 1)
/// ```
///
/// The implementation is comparison-based: items are stored and sorted
/// directly, so no hashing, seeds, or geometry compatibility are involved —
/// merging only requires equal `k`. The original paper additionally tapers
/// level capacities and samples at the bottom level; this implementation
/// uses a constant capacity per level, the same simplification as the
/// production KLL in Apache DataSketches.
///
/// # References
///
/// - Zohar Karnin, Kevin Lang, and Edo Liberty, "Optimal Quantile
///   Approximation in Streams", FOCS 2016. <https://doi.org/10.1109/FOCS.2016.17>
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KllSketch<T> {
    levels: Vec<Vec<T>>,
    k: usize,
    count: u64,
    rng: XorShift64,
}

impl<T> KllSketch<T> {
    /// Creates an empty KLL sketch with capacity `k` per level and
    /// compaction seed zero.
    ///
    /// # Panics
    ///
    /// Panics if `k < 8`.
    pub fn new(k: usize) -> Self {
        Self::with_seed(k, 0)
    }

    /// Creates an empty KLL sketch with an explicit compaction seed, making
    /// compaction choices deterministic.
    ///
    /// # Panics
    ///
    /// Panics if `k < 8`.
    pub fn with_seed(k: usize, seed: u64) -> Self {
        assert!(k >= 8, "KLL capacity must be at least 8");
        Self {
            levels: vec![Vec::new()],
            k,
            count: 0,
            rng: XorShift64::new(seed),
        }
    }

    /// Returns the per-level capacity.
    pub const fn k(&self) -> usize {
        self.k
    }

    /// Returns the number of inserted items.
    pub const fn count(&self) -> u64 {
        self.count
    }

    /// Returns the number of compactor levels currently in use.
    pub fn levels(&self) -> usize {
        self.levels.len()
    }

    /// Returns the number of bytes held by the stored values.
    pub fn storage_bytes(&self) -> usize {
        self.levels.iter().map(Vec::len).sum::<usize>() * size_of::<T>()
    }

    /// Clears the sketch back to empty, keeping the compaction seed stream.
    pub fn clear(&mut self) {
        self.levels.truncate(1);
        self.levels[0].clear();
        self.count = 0;
    }

    /// Compacts level `h`, promoting every other value to level `h + 1`
    /// with doubled weight. An odd-length buffer keeps its largest value
    /// behind, preserving the total weight exactly.
    fn compact(&mut self, h: usize)
    where
        T: Ord,
    {
        let mut buffer = core::mem::take(&mut self.levels[h]);
        buffer.sort_unstable();
        let tail = if buffer.len() % 2 == 1 {
            buffer.pop()
        } else {
            None
        };

        let parity = self.rng.next_index(2);
        let mut promoted = Vec::with_capacity(buffer.len() / 2);
        for (i, item) in buffer.into_iter().enumerate() {
            if i % 2 == parity {
                promoted.push(item);
            }
        }

        self.levels[h] = match tail {
            Some(item) => vec![item],
            None => Vec::new(),
        };
        if self.levels.len() == h + 1 {
            self.levels.push(Vec::new());
        }
        self.levels[h + 1].append(&mut promoted);
        if self.levels[h + 1].len() > self.k {
            self.compact(h + 1);
        }
    }

    /// Collects `(weight, value)` pairs across all levels, sorted by value.
    fn weighted_items(&self) -> Vec<(u64, &T)> {
        let mut items = Vec::new();
        for (h, level) in self.levels.iter().enumerate() {
            // Level heights stay far below 64 in practice; saturate rather
            // than overflow if a pathological stream ever gets there.
            let weight = 1_u64 << h.min(63);
            items.extend(level.iter().map(|item| (weight, item)));
        }
        items
    }
}

impl<T: Ord> KllSketch<T> {
    /// Inserts `item`, compacting upward when level 0 overflows.
    pub fn insert_item(&mut self, item: &T)
    where
        T: Clone,
    {
        self.levels[0].push(item.clone());
        self.count += 1;
        if self.levels[0].len() > self.k {
            self.compact(0);
        }
    }

    /// Returns an estimate of the `q`-quantile: the value whose normalized
    /// rank is `q`, or `None` when the sketch is empty.
    ///
    /// # Panics
    ///
    /// Panics if `q` is not in `0.0..=1.0`.
    pub fn quantile(&self, q: f64) -> Option<T>
    where
        T: Clone,
    {
        assert!(
            (0.0..=1.0).contains(&q),
            "quantile must be in 0.0..=1.0"
        );
        if self.count == 0 {
            return None;
        }

        let mut items = self.weighted_items();
        items.sort_by(|a, b| a.1.cmp(b.1));
        let target = q * (self.count - 1) as f64;
        let mut cumulative = 0_u64;
        for (weight, item) in &items {
            cumulative += weight;
            if (cumulative as f64) > target {
                return Some((*item).clone());
            }
        }
        Some((*items.last().expect("non-empty sketch").1).clone())
    }

    /// Returns the estimated normalized rank of `item` in `0.0..=1.0`: the
    /// weighted fraction of inserted values smaller than `item`.
    pub fn rank(&self, item: &T) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let mut smaller = 0_u64;
        for (h, level) in self.levels.iter().enumerate() {
            let weight = 1_u64 << h.min(63);
            for value in level {
                if value < item {
                    smaller += weight;
                }
            }
        }
        smaller as f64 / self.count as f64
    }

    /// Returns an estimate of the median.
    pub fn median(&self) -> Option<T>
    where
        T: Clone,
    {
        self.quantile(0.5)
    }
}

impl<T> Sketch for KllSketch<T> {
    fn clear(&mut self) {
        self.clear();
    }

    fn len_hint(&self) -> Option<u64> {
        Some(self.count)
    }

    fn storage_bytes(&self) -> usize {
        self.storage_bytes()
    }
}

impl<T> Insert<T> for KllSketch<T>
where
    T: Ord + Clone,
{
    type Err = Infallible;

    fn insert(&mut self, item: &T) -> Result<(), Self::Err> {
        self.insert_item(item);
        Ok(())
    }
}

impl<T: Ord + Clone> Merge for KllSketch<T> {
    /// Merges level by level: items of equal weight are combined and any
    /// overflowing level compacts upward. Only `k` must match — compaction
    /// randomness is internal and does not affect merge compatibility.
    fn merge_from(&mut self, other: &Self) -> Result<(), MergeError> {
        if self.k != other.k {
            return Err(MergeError::GeometryMismatch);
        }
        for h in 0..other.levels.len() {
            if self.levels.len() == h {
                self.levels.push(Vec::new());
            }
            if other.levels[h].is_empty() {
                continue;
            }
            let mut incoming = other.levels[h].clone();
            self.levels[h].append(&mut incoming);
            if self.levels[h].len() > self.k {
                self.compact(h);
            }
        }
        self.count += other.count;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::KllSketch;
    use crate::error::MergeError;
    use crate::traits::{Insert, Merge, Sketch};

    #[test]
    fn empty_sketch_answers_none_and_zero() {
        let sketch = KllSketch::<u64>::new(8);
        assert_eq!(sketch.quantile(0.5), None);
        assert_eq!(sketch.rank(&42), 0.0);
        assert_eq!(Sketch::len_hint(&sketch), Some(0));
    }

    #[test]
    fn up_to_capacity_results_are_exact() {
        // No compaction happens while n <= k, so quantiles are exact.
        let mut sketch = KllSketch::new(8);
        for i in [5_u64, 1, 8, 3, 9, 2, 7, 4] {
            sketch.insert_item(&i);
        }
        assert_eq!(sketch.quantile(0.0), Some(1));
        assert_eq!(sketch.median(), Some(4));
        assert_eq!(sketch.quantile(1.0), Some(9));
        assert_eq!(sketch.rank(&4), 3.0 / 8.0);
    }

    #[test]
    fn compaction_preserves_total_count() {
        let mut sketch = KllSketch::with_seed(8, 3);
        for i in 0..1_000_u64 {
            sketch.insert_item(&i);
        }
        assert_eq!(sketch.count(), 1_000);
        assert!(sketch.levels() > 1);
    }

    #[test]
    fn same_seed_gives_identical_results() {
        let mut left = KllSketch::with_seed(16, 9);
        let mut right = KllSketch::with_seed(16, 9);
        for i in 0..10_000_u64 {
            let value = (i * 2_654_435_761) % 10_000;
            left.insert_item(&value);
            right.insert_item(&value);
        }
        for q in [0.1, 0.5, 0.9] {
            assert_eq!(left.quantile(q), right.quantile(q));
        }
    }

    #[test]
    fn merge_combines_streams_and_validates_k() {
        let mut left = KllSketch::with_seed(16, 1);
        let mut right = KllSketch::with_seed(16, 2);
        for i in 0..5_000_u64 {
            left.insert_item(&i);
            right.insert_item(&(5_000 + i));
        }

        left.merge_from(&right).unwrap();
        assert_eq!(left.count(), 10_000);
        // Compaction may drop the exact extremes at doubled weight, so the
        // endpoints and median are only guaranteed up to the rank error.
        assert!(left.quantile(0.0).unwrap() <= 2_500);
        assert!(left.quantile(1.0).unwrap() >= 7_500);
        let median = left.quantile(0.5).unwrap();
        assert!((2_500..=7_500).contains(&median), "median {median}");

        let other_k = KllSketch::<u64>::new(32);
        assert_eq!(left.merge_from(&other_k), Err(MergeError::GeometryMismatch));
    }

    #[test]
    fn capability_traits_work() {
        let mut sketch = KllSketch::<u64>::new(8);
        Insert::<u64>::insert(&mut sketch, &7).unwrap();
        assert_eq!(Sketch::len_hint(&sketch), Some(1));
    }

    #[test]
    #[should_panic(expected = "KLL capacity must be at least 8")]
    fn capacity_is_validated() {
        KllSketch::<u64>::new(4);
    }
}
