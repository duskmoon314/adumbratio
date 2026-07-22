//! Top-k heavy hitters over a Count-Min Sketch.

use core::convert::Infallible;
use core::hash::{BuildHasher, Hash};

use alloc::vec::Vec;

use crate::error::MergeError;
use crate::sketch::CountMinSketch;
use crate::traits::{Insert, Merge, Sketch};

/// A heavy-hitters companion that tracks the top-`k` items over a
/// [`CountMinSketch`].
///
/// Count-Min gives point estimates for any item but cannot enumerate the
/// heavy ones. The standard companion structure keeps a small candidate set
/// alongside the sketch: every inserted item updates the sketch, and the
/// candidate set keeps the `k` distinct items with the largest estimates
/// seen so far. Because Count-Min estimates grow monotonically, replacing
/// the weakest candidate whenever a stronger one appears is enough to
/// recover the true heavy hitters on skewed streams.
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
/// Candidate membership is checked by a linear scan, so one insert costs
/// `O(k)` comparisons on top of the sketch update — fine for analytics
/// companions, not for line-rate data planes.
///
/// # References
///
/// - Graham Cormode and Marios Hadjieleftheriou, "Finding Frequent Items in
///   Data Streams", PVLDB 2008. <https://doi.org/10.14778/1454159.1454225>
/// - Graham Cormode and S. Muthukrishnan, "An Improved Data Stream Summary:
///   The Count-Min Sketch and its Applications", Journal of Algorithms, 2005.
///   <https://doi.org/10.1016/j.jalgor.2003.12.001>
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TopK<T, S = crate::hash::DefaultBuildHasher> {
    sketch: CountMinSketch<32, S>,
    candidates: Vec<T>,
    k: usize,
}

impl<T> TopK<T, crate::hash::DefaultBuildHasher> {
    /// Creates a top-k tracker whose sketch is solved from `epsilon` and
    /// `delta`, with seed zero.
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

    /// Creates a top-k tracker with an explicit hash seed.
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
}

impl<T, S> TopK<T, S> {
    /// Creates a top-k tracker over an existing Count-Min Sketch.
    ///
    /// # Panics
    ///
    /// Panics if `k` is zero.
    pub fn from_sketch(k: usize, sketch: CountMinSketch<32, S>) -> Self {
        assert!(k > 0, "top-k tracking needs k >= 1");
        Self {
            sketch,
            candidates: Vec::new(),
            k,
        }
    }

    /// Returns the number of tracked items.
    pub const fn k(&self) -> usize {
        self.k
    }

    /// Returns the underlying sketch.
    pub const fn sketch(&self) -> &CountMinSketch<32, S> {
        &self.sketch
    }

    /// Returns the current candidate set, in no particular order.
    pub fn candidates(&self) -> &[T] {
        &self.candidates
    }

    /// Returns the total number of inserted events.
    pub const fn total_count(&self) -> u64 {
        self.sketch.total_count()
    }

    /// Clears the sketch and the candidate set.
    pub fn clear(&mut self) {
        self.sketch.clear();
        self.candidates.clear();
    }

    /// Returns the byte length of the sketch storage (excluding candidates).
    pub fn storage_bytes(&self) -> usize {
        self.sketch.storage_bytes()
    }
}

impl<T, S> TopK<T, S>
where
    T: Hash + Eq,
    S: BuildHasher,
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
                let estimate = self.sketch.estimate_item(item);
                (item.clone(), estimate)
            })
            .collect();
        items.sort_by(|a, b| b.1.cmp(&a.1));
        items
    }

    /// Inserts one occurrence of `item` and maintains the candidate set.
    pub fn insert_item(&mut self, item: &T)
    where
        T: Clone,
    {
        self.sketch.insert_item(item);

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
            .map(|(i, candidate)| (i, self.sketch.estimate_item(candidate)))
            .min_by_key(|&(_, estimate)| estimate)
            .expect("candidate set is full and non-empty");
        if self.sketch.estimate_item(item) > weakest_estimate {
            self.candidates[weakest] = item.clone();
        }
    }

    /// Estimates the frequency of `item` from the underlying sketch.
    pub fn estimate_item(&self, item: &T) -> u64 {
        self.sketch.estimate_item(item)
    }
}

impl<T, S> Sketch for TopK<T, S> {
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

impl<T, S> Insert<T> for TopK<T, S>
where
    T: Hash + Eq + Clone,
    S: BuildHasher,
{
    type Err = Infallible;

    fn insert(&mut self, item: &T) -> Result<(), Self::Err> {
        self.insert_item(item);
        Ok(())
    }
}

impl<T, S> Merge for TopK<T, S>
where
    T: Hash + Eq + Clone,
    S: BuildHasher,
{
    /// Merges the underlying sketches and re-selects the top-k candidates
    /// from the union of both candidate sets.
    fn merge_from(&mut self, other: &Self) -> Result<(), MergeError> {
        if self.k != other.k {
            return Err(MergeError::GeometryMismatch);
        }
        self.sketch.merge_from(&other.sketch)?;

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
        self.candidates
            .sort_by(|a, b| {
                self.sketch
                    .estimate_item(b)
                    .cmp(&self.sketch.estimate_item(a))
            });
        self.candidates.truncate(self.k);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
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
    fn capability_traits_work() {
        let mut top = TopK::new(3, 0.001, 0.01);
        Insert::<u64>::insert(&mut top, &7).unwrap();
        assert!(top.estimate_item(&7) >= 1);
    }
}
