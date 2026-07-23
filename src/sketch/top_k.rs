//! Top-k heavy hitters over a frequency sketch.

use core::convert::Infallible;
use core::hash::{BuildHasher, Hash};

use alloc::vec::Vec;

use crate::error::MergeError;
use crate::sketch::CountMinSketch;
#[cfg(any(feature = "std", feature = "libm"))]
use crate::sketch::CountSketch;
use crate::traits::{Estimator, Insert, Merge, Sketch};

/// A heavy-hitters companion that tracks the top-`k` items over a
/// frequency [`Estimator`] — a [`CountMinSketch`] (default) or a
/// [`CountSketch`].
///
/// Frequency sketches give point estimates for any item but cannot
/// enumerate the heavy ones. The standard companion structure keeps a small
/// candidate set alongside the sketch: every inserted item updates the
/// sketch, and the candidate set keeps the `k` distinct items with the
/// largest estimates seen so far. Because estimates grow monotonically,
/// replacing the weakest candidate whenever a stronger one appears is
/// enough to recover the true heavy hitters on skewed streams.
///
/// ```text
/// insert("x")
///      |
///      v
///   sketch.estimate("x") += 1
///      |
///      +--> already a candidate?        done
///      +--> fewer than k candidates?    push
///      +--> estimate > weakest?         replace weakest
///
/// top_k() = candidates sorted by their current estimates
/// ```
///
/// With `std`, a hash index plus a lazy versioned min-heap make inserts
/// `O(log k)`; the `no_std` fallback keeps an `O(k)` linear scan.
///
/// # References
///
/// - Graham Cormode and Marios Hadjieleftheriou, "Finding Frequent Items in
///   Data Streams", PVLDB 2008. <https://doi.org/10.14778/1454159.1454225>
/// - Graham Cormode and S. Muthukrishnan, "An Improved Data Stream Summary:
///   The Count-Min Sketch and its Applications", Journal of Algorithms, 2005.
///   <https://doi.org/10.1016/j.jalgor.2003.12.001>
/// - Moses Charikar, Kevin Chen, and Martin Farach-Colton, "Finding Frequent
///   Items in Data Streams", Theoretical Computer Science, 2004.
///   <https://doi.org/10.1016/j.tcs.2003.10.024>
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "T: serde::Serialize + Eq + core::hash::Hash, S: serde::Serialize, E: serde::Serialize",
        deserialize = "T: serde::Deserialize<'de> + Eq + core::hash::Hash, S: serde::Deserialize<'de>, E: serde::Deserialize<'de>"
    ))
)]
pub struct TopK<T, S = crate::hash::DefaultBuildHasher, E = CountMinSketch<32, S>> {
    sketch: E,
    candidates: Vec<T>,
    k: usize,
    /// Anchors the hasher type parameter (only used through `E`'s default).
    #[cfg_attr(feature = "serde", serde(skip))]
    marker: core::marker::PhantomData<S>,
    /// Maps candidate items to their slot in `candidates` (`std` fast path).
    #[cfg(feature = "std")]
    index: std::collections::HashMap<T, usize>,
    /// Min-heap of `(estimate, version, slot)`; entries go stale as
    /// estimates grow and are discarded lazily on eviction.
    #[cfg(feature = "std")]
    heap: std::collections::BinaryHeap<std::cmp::Reverse<(u64, u64, usize)>>,
    /// Per-slot version counter distinguishing live heap entries from stale
    /// ones; reset when a slot changes identity.
    #[cfg(feature = "std")]
    versions: Vec<u64>,
}

impl<T> TopK<T, crate::hash::DefaultBuildHasher> {
    /// Creates a top-k tracker whose Count-Min backend is solved from
    /// `epsilon` and `delta`, with seed zero.
    ///
    /// Available with the `std` or `libm` feature.
    ///
    /// # Panics
    ///
    /// Panics if `k` is zero or if the error parameters are invalid (see
    /// [`CountMinSketch::with_error`]).
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn new(k: usize, epsilon: f64, delta: f64) -> Self {
        Self::with_seed(k, epsilon, delta, 0)
    }

    /// Creates a top-k tracker over Count-Min with an explicit hash seed.
    ///
    /// Available with the `std` or `libm` feature.
    ///
    /// # Panics
    ///
    /// Panics if `k` is zero or if the error parameters are invalid.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn with_seed(k: usize, epsilon: f64, delta: f64, seed: u64) -> Self {
        Self::from_sketch(k, CountMinSketch::with_error_and_seed(epsilon, delta, seed))
    }

    /// Creates a top-k tracker backed by a **Count Sketch** solved from
    /// `epsilon` and `delta`, with seed zero — for consumers whose window
    /// state is already Count Sketch, or who want median-of-rows
    /// (unbiased) heavy-hitter estimates.
    ///
    /// Available with the `std` or `libm` feature.
    ///
    /// # Panics
    ///
    /// Panics if `k` is zero or if the error parameters are invalid.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn with_count_sketch(
        k: usize,
        epsilon: f64,
        delta: f64,
    ) -> TopK<T, crate::hash::DefaultBuildHasher, CountSketch> {
        Self::with_count_sketch_and_seed(k, epsilon, delta, 0)
    }

    /// Creates a top-k tracker backed by a Count Sketch with an explicit
    /// hash seed.
    ///
    /// Available with the `std` or `libm` feature.
    ///
    /// # Panics
    ///
    /// Panics if `k` is zero or if the error parameters are invalid.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn with_count_sketch_and_seed(
        k: usize,
        epsilon: f64,
        delta: f64,
        seed: u64,
    ) -> TopK<T, crate::hash::DefaultBuildHasher, CountSketch> {
        TopK::<T, _, CountSketch>::from_sketch(k, CountSketch::with_error_and_seed(epsilon, delta, seed))
    }
}

impl<T, S, E> TopK<T, S, E> {
    /// Creates a top-k tracker over an existing frequency sketch.
    ///
    /// # Panics
    ///
    /// Panics if `k` is zero.
    pub fn from_sketch(k: usize, sketch: E) -> Self {
        assert!(k > 0, "top-k tracking needs k >= 1");
        Self {
            sketch,
            candidates: Vec::new(),
            k,
            marker: core::marker::PhantomData,
            #[cfg(feature = "std")]
            index: std::collections::HashMap::new(),
            #[cfg(feature = "std")]
            heap: std::collections::BinaryHeap::new(),
            #[cfg(feature = "std")]
            versions: Vec::new(),
        }
    }

    /// Returns the number of tracked items.
    pub const fn k(&self) -> usize {
        self.k
    }

    /// Returns the underlying sketch.
    pub const fn sketch(&self) -> &E {
        &self.sketch
    }

    /// Returns the current candidate set, in no particular order.
    pub fn candidates(&self) -> &[T] {
        &self.candidates
    }
}

impl<T, S, E> TopK<T, S, E>
where
    E: Estimator<T>,
{
    /// Returns the total number of inserted events.
    pub fn total_count(&self) -> u64 {
        Estimator::total(&self.sketch)
    }

    /// Clears the sketch and the candidate set.
    pub fn clear(&mut self) {
        Sketch::clear(&mut self.sketch);
        self.candidates.clear();
        #[cfg(feature = "std")]
        {
            self.index.clear();
            self.heap.clear();
            self.versions.clear();
        }
    }

    /// Returns the byte length of the sketch storage (excluding candidates).
    pub fn storage_bytes(&self) -> usize {
        Sketch::storage_bytes(&self.sketch)
    }
}

impl<T, S, E> TopK<T, S, E>
where
    T: Hash + Eq,
    S: BuildHasher,
    E: Estimator<T>,
{
    /// Returns the candidates with their current estimates, heaviest first.
    pub fn top_k(&self) -> Vec<(T, u64)>
    where
        T: Clone,
    {
        let mut items: Vec<_> = self
            .candidates
            .iter()
            .map(|item| {
                let estimate = Estimator::estimate(&self.sketch, item);
                (item.clone(), estimate)
            })
            .collect();
        items.sort_by_key(|item| core::cmp::Reverse(item.1));
        items
    }

    /// Inserts one occurrence of `item` and maintains the candidate set
    /// (`std` fast path: hash index + lazy min-heap, `O(log k)`).
    #[cfg(feature = "std")]
    pub fn insert_item(&mut self, item: &T)
    where
        T: Clone,
    {
        use std::cmp::Reverse;

        Estimator::insert_count(&mut self.sketch, item, 1);

        if let Some(&slot) = self.index.get(item) {
            let estimate = Estimator::estimate(&self.sketch, item);
            self.versions[slot] += 1;
            self.heap.push(Reverse((estimate, self.versions[slot], slot)));
            return;
        }
        if self.candidates.len() < self.k {
            let slot = self.candidates.len();
            let estimate = Estimator::estimate(&self.sketch, item);
            self.candidates.push(item.clone());
            self.index.insert(item.clone(), slot);
            self.versions.push(0);
            self.heap.push(Reverse((estimate, 0, slot)));
            return;
        }

        // Find the freshest minimum, discarding stale heap entries.
        let (min_estimate, victim) = loop {
            let Reverse((estimate, version, slot)) =
                self.heap.pop().expect("heap is non-empty when the candidate set is full");
            if self.versions[slot] == version {
                break (estimate, slot);
            }
        };
        let estimate = Estimator::estimate(&self.sketch, item);
        if estimate > min_estimate {
            self.index.remove(&self.candidates[victim]);
            self.index.insert(item.clone(), victim);
            self.candidates[victim] = item.clone();
            self.versions[victim] = 0;
            self.heap.push(Reverse((estimate, 0, victim)));
        } else {
            // Not evicting after all: return the minimum entry to the heap.
            self.heap
                .push(Reverse((min_estimate, self.versions[victim], victim)));
        }
    }

    /// Inserts one occurrence of `item` and maintains the candidate set
    /// (`no_std` fallback: `O(k)` linear scan).
    #[cfg(not(feature = "std"))]
    pub fn insert_item(&mut self, item: &T)
    where
        T: Clone,
    {
        Estimator::insert_count(&mut self.sketch, item, 1);

        if self.candidates.iter().any(|candidate| candidate == item) {
            return;
        }
        if self.candidates.len() < self.k {
            self.candidates.push(item.clone());
            return;
        }

        // Replace the weakest candidate when the new item is heavier. Since
        // estimates only grow, the weakest candidate is the one with the
        // smallest current estimate.
        let (weakest, weakest_estimate) = self
            .candidates
            .iter()
            .enumerate()
            .map(|(i, candidate)| (i, Estimator::estimate(&self.sketch, candidate)))
            .min_by_key(|&(_, estimate)| estimate)
            .expect("candidate set is full and non-empty");
        if Estimator::estimate(&self.sketch, item) > weakest_estimate {
            self.candidates[weakest] = item.clone();
        }
    }

    /// Estimates the frequency of `item` from the underlying sketch.
    pub fn estimate_item(&self, item: &T) -> u64 {
        Estimator::estimate(&self.sketch, item)
    }
}

impl<T, S, E> Sketch for TopK<T, S, E>
where
    E: Estimator<T>,
{
    fn clear(&mut self) {
        self.clear();
    }

    fn len_hint(&self) -> Option<u64> {
        Some(self.total_count())
    }

    fn storage_bytes(&self) -> usize {
        self.storage_bytes()
    }
}

impl<T, S, E> Insert<T> for TopK<T, S, E>
where
    T: Hash + Eq + Clone,
    S: BuildHasher,
    E: Estimator<T>,
{
    type Err = Infallible;

    fn insert(&mut self, item: &T) -> Result<(), Self::Err> {
        self.insert_item(item);
        Ok(())
    }
}

impl<T, S, E> Merge for TopK<T, S, E>
where
    T: Hash + Eq + Clone,
    S: BuildHasher,
    E: Estimator<T>,
{
    /// Merges the underlying sketches and re-selects the top-k candidates
    /// from the union of both candidate sets.
    fn merge_from(&mut self, other: &Self) -> Result<(), MergeError> {
        if self.k != other.k {
            return Err(MergeError::GeometryMismatch);
        }
        Merge::merge_from(&mut self.sketch, &other.sketch)?;

        let union: Vec<T> = self
            .candidates
            .iter()
            .chain(other.candidates.iter())
            .cloned()
            .fold(Vec::new(), |mut acc, item| {
                if !acc.contains(&item) {
                    acc.push(item);
                }
                acc
            });
        self.candidates = union;
        // Trim to the k strongest under the merged sketch.
        self.candidates.sort_by(|a, b| {
            Estimator::estimate(&self.sketch, b).cmp(&Estimator::estimate(&self.sketch, a))
        });
        self.candidates.truncate(self.k);

        // Rebuild the fast-path index and heap from the new candidate set.
        #[cfg(feature = "std")]
        {
            use std::cmp::Reverse;

            self.index.clear();
            self.heap.clear();
            self.versions.clear();
            for (slot, item) in self.candidates.iter().enumerate() {
                self.index.insert(item.clone(), slot);
                self.versions.push(0);
                let estimate = Estimator::estimate(&self.sketch, item);
                self.heap.push(Reverse((estimate, 0, slot)));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::TopK;
    use crate::traits::{Insert, Merge, Sketch};

    fn skewed_stream() -> Vec<u64> {
        // Item 0 dominates, then a fast-decaying tail.
        let mut stream = Vec::new();
        for item in 0..100_u64 {
            for _ in 0..(10_000 / (item + 1)) {
                stream.push(item);
            }
        }
        stream
    }

    #[test]
    fn heavy_items_are_tracked_in_order() {
        let mut top = TopK::new(10, 0.001, 0.01);
        for item in skewed_stream() {
            top.insert_item(&item);
        }

        let ranked = top.top_k();
        assert_eq!(ranked.len(), 10);
        assert_eq!(ranked[0].0, 0);
        assert!(ranked[0].1 >= 9_999);
        // The ten heaviest items of this stream are 0..=9.
        for (rank, (item, _)) in ranked.iter().enumerate() {
            assert_eq!(*item, rank as u64);
        }
    }

    #[test]
    fn candidates_stay_capped_and_estimates_delegate() {
        let mut top = TopK::new(5, 0.001, 0.01);
        for item in skewed_stream() {
            top.insert_item(&item);
        }
        assert_eq!(top.candidates().len(), 5);
        assert!(top.estimate_item(&0_u64) >= 9_999);
        assert_eq!(Sketch::len_hint(&top), Some(top.total_count()));
    }

    #[test]
    fn merge_recovers_union_heavy_hitters() {
        let mut left = TopK::new(5, 0.001, 0.01);
        let mut right = TopK::new(5, 0.001, 0.01);
        let stream = skewed_stream();
        for (i, item) in stream.iter().enumerate() {
            if i % 2 == 0 {
                left.insert_item(item);
            } else {
                right.insert_item(item);
            }
        }

        left.merge_from(&right).unwrap();
        let ranked = left.top_k();
        assert_eq!(ranked[0].0, 0);
        for (rank, (item, _)) in ranked.iter().enumerate() {
            assert_eq!(*item, rank as u64);
        }
    }

    #[test]
    fn eviction_stays_correct_with_stale_heap_entries() {
        // Heavy updates to a few candidates create many stale heap entries;
        // eviction must still pick the true weakest candidate.
        let mut top = TopK::new(3, 0.001, 0.01);
        for _ in 0..100 {
            top.insert_item(&10_u64);
            top.insert_item(&11_u64);
            top.insert_item(&12_u64);
        }
        // A new item heavier than the weakest (12 is weakest with 100 < 101?).
        for _ in 0..101 {
            top.insert_item(&13_u64);
        }
        let ranked = top.top_k();
        assert!(ranked.iter().any(|(item, _)| *item == 13));
        // The weakest of the original three (all tied at 100) may survive as
        // any two of them; the key assertion is the candidate count and that
        // 13 displaced one of the originals.
        assert_eq!(ranked.len(), 3);
        assert!(
            ranked
                .iter()
                .filter(|(item, _)| [10, 11, 12].contains(item))
                .count()
                == 2
        );
    }

    #[test]
    fn count_sketch_backend_recovers_heavy_items() {
        let mut top = TopK::<u64>::with_count_sketch(10, 0.01, 0.01);
        for item in skewed_stream() {
            top.insert_item(&item);
        }

        let ranked = top.top_k();
        assert_eq!(ranked.len(), 10);
        assert_eq!(ranked[0].0, 0);
        assert!(ranked[0].1 >= 9_000);
        // Count Sketch is unbiased; heavy ranks come back in order on this
        // strongly skewed stream.
        for (rank, (item, _)) in ranked.iter().enumerate() {
            assert_eq!(*item, rank as u64);
        }
        assert!(top.estimate_item(&0_u64) >= 9_000);
        assert_eq!(top.total_count(), top.sketch().total_count());
    }

    #[test]
    fn capability_traits_work() {
        let mut top = TopK::new(3, 0.001, 0.01);
        Insert::<u64>::insert(&mut top, &7).unwrap();
        assert!(top.estimate_item(&7) >= 1);
    }
}
