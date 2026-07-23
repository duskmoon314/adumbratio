//! DDSketch: a fully-mergeable quantile sketch with relative-error
//! guarantees.

#[cfg(any(feature = "std", feature = "libm"))]
use core::convert::Infallible;

use alloc::vec::Vec;

use crate::error::MergeError;
#[cfg(any(feature = "std", feature = "libm"))]
use crate::float;
#[cfg(any(feature = "std", feature = "libm"))]
use crate::traits::Insert;
use crate::traits::{Merge, Sketch};

/// A DDSketch quantile sketch with *relative* error guarantees.
///
/// Where KLL promises an absolute rank error `eps * N`, DDSketch promises
/// that every quantile estimate is within a factor `1 +/- alpha` of the
/// true quantile *value*: the p99 latency is as accurate, relatively, as
/// the median. The price is restriction to positive values and logarithmic
/// space growth with the value range.
///
/// Values map to logarithmic buckets with ratio
/// `gamma = (1 + alpha) / (1 - alpha)`: bucket `i` covers
/// `(gamma^(i-1), gamma^i]`, so any value taken as a bucket's estimate is
/// at most a factor `alpha` off. Quantiles read out per-bucket estimates
/// `2 * gamma^i / (gamma + 1)`.
///
/// ```text
/// insert(x): counts[ceil(log_gamma(x))] += 1
///
/// quantile(q): first bucket whose cumulative count reaches q * N,
///              estimated as 2 * gamma^i / (gamma + 1)
/// ```
///
/// Buckets are added on demand, so the sketch works over an unbounded
/// value range. Merging adds bucket counts pairwise and is exact — the
/// paper's headline property for distributed aggregation.
///
/// # References
///
/// - Charles Masson, Jee E. Rim, and Homin K. Lee, "DDSketch: A Fast and
///   Fully-Mergeable Quantile Sketch with Relative-Error Guarantees",
///   PVLDB 12(12), 2019. <https://doi.org/10.14778/3352063.3352135>
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DdSketch {
    buckets: Vec<(i32, u64)>,
    log_gamma: f64,
    count: u64,
}

impl DdSketch {
    /// Creates a DDSketch with relative-error target `alpha` in
    /// `0.0..=1.0`: every quantile estimate is within a factor
    /// `1 +/- alpha` of the true value.
    ///
    /// Available with the `std` or `libm` feature.
    ///
    /// # Panics
    ///
    /// Panics if `alpha` is not finite and in `0.0..=1.0`.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn new(alpha: f64) -> Self {
        assert!(
            alpha.is_finite() && alpha > 0.0 && alpha < 1.0,
            "alpha must be finite and in 0.0..1.0"
        );
        let gamma = (1.0 + alpha) / (1.0 - alpha);
        Self::from_parts(float::ln(gamma))
    }

    /// Creates a DDSketch from an explicit `ln(gamma)`, available without
    /// float math for callers that precompute it.
    pub fn from_parts(log_gamma: f64) -> Self {
        assert!(
            log_gamma.is_finite() && log_gamma > 0.0,
            "log_gamma must be finite and positive"
        );
        Self {
            buckets: Vec::new(),
            log_gamma,
            count: 0,
        }
    }

    /// Returns `gamma = e^log_gamma`, the bucket ratio.
    ///
    /// Available with the `std` or `libm` feature.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn gamma(&self) -> f64 {
        float::exp(self.log_gamma)
    }

    /// Returns the number of inserted values.
    pub const fn count(&self) -> u64 {
        self.count
    }

    /// Returns the number of non-empty buckets.
    pub fn bucket_count(&self) -> usize {
        self.buckets.len()
    }

    /// Returns the non-empty buckets as `(index, count)` pairs, sorted.
    pub fn buckets(&self) -> &[(i32, u64)] {
        &self.buckets
    }

    /// Returns the byte length of the bucket storage.
    pub fn storage_bytes(&self) -> usize {
        self.buckets.len() * (size_of::<i32>() + size_of::<u64>())
    }

    /// Clears all buckets.
    pub fn clear(&mut self) {
        self.buckets.clear();
        self.count = 0;
    }

    /// Maps a positive value to its bucket index.
    ///
    /// Available with the `std` or `libm` feature.
    #[cfg(any(feature = "std", feature = "libm"))]
    fn bucket_index(&self, value: f64) -> i32 {
        debug_assert!(value > 0.0, "DDSketch only accepts positive values");
        float::ceil(float::ln(value) / self.log_gamma) as i32
    }

    /// The estimated value of a bucket, `2 * gamma^i / (gamma + 1)`.
    ///
    /// Available with the `std` or `libm` feature.
    #[cfg(any(feature = "std", feature = "libm"))]
    fn bucket_value(&self, index: i32) -> f64 {
        let gamma = self.gamma();
        2.0 * float::powf(gamma, f64::from(index)) / (gamma + 1.0)
    }

    /// Inserts `count` occurrences into the bucket at `index`, keeping the
    /// list sorted.
    fn insert_at(&mut self, index: i32, count: u64) {
        match self.buckets.binary_search_by_key(&index, |&(i, _)| i) {
            Ok(position) => self.buckets[position].1 += count,
            Err(position) => self.buckets.insert(position, (index, count)),
        }
    }
}

impl Sketch for DdSketch {
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

#[cfg(any(feature = "std", feature = "libm"))]
impl Insert<f64> for DdSketch {
    type Err = Infallible;

    fn insert(&mut self, item: &f64) -> Result<(), Self::Err> {
        self.insert_item(item);
        Ok(())
    }
}

impl DdSketch {
    /// Inserts one positive value.
    ///
    /// Available with the `std` or `libm` feature.
    ///
    /// # Panics
    ///
    /// Panics if `value` is not positive and finite.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn insert_item(&mut self, value: &f64) {
        assert!(
            value.is_finite() && *value > 0.0,
            "DDSketch only accepts positive finite values"
        );
        let index = self.bucket_index(*value);
        self.insert_at(index, 1);
        self.count += 1;
    }

    /// Returns the estimated `q`-quantile value, or `None` when empty.
    ///
    /// Available with the `std` or `libm` feature.
    ///
    /// # Panics
    ///
    /// Panics if `q` is not in `0.0..=1.0`.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn quantile(&self, q: f64) -> Option<f64> {
        assert!((0.0..=1.0).contains(&q), "quantile must be in 0.0..=1.0");
        if self.count == 0 {
            return None;
        }
        let target = float::ceil(q * (self.count - 1) as f64) as u64;
        let mut cumulative = 0_u64;
        for &(index, count) in &self.buckets {
            cumulative += count;
            if cumulative > target {
                return Some(self.bucket_value(index));
            }
        }
        let &(index, _) = self.buckets.last().expect("non-empty sketch");
        Some(self.bucket_value(index))
    }

    /// Returns the estimated normalized rank of `value` in `0.0..=1.0`.
    ///
    /// Available with the `std` or `libm` feature.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn rank(&self, value: &f64) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let index = self.bucket_index(*value);
        let smaller: u64 = self
            .buckets
            .iter()
            .take_while(|&&(i, _)| i < index)
            .map(|&(_, count)| count)
            .sum();
        smaller as f64 / self.count as f64
    }

    /// Returns the estimated median value.
    ///
    /// Available with the `std` or `libm` feature.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn median(&self) -> Option<f64> {
        self.quantile(0.5)
    }
}

impl Merge for DdSketch {
    /// Merges by adding bucket counts pairwise — exact, no seed or hash
    /// involved. The bucket ratio `gamma` must match.
    fn merge_from(&mut self, other: &Self) -> Result<(), MergeError> {
        if self.log_gamma.to_bits() != other.log_gamma.to_bits() {
            return Err(MergeError::GeometryMismatch);
        }
        for &(index, count) in &other.buckets {
            self.insert_at(index, count);
        }
        self.count += other.count;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::DdSketch;
    use crate::error::MergeError;
    use crate::traits::{Insert, Merge, Sketch};

    #[test]
    fn empty_sketch_answers_none_and_zero() {
        let sketch = DdSketch::new(0.05);
        assert_eq!(sketch.quantile(0.5), None);
        assert_eq!(sketch.rank(&1.0), 0.0);
        assert_eq!(Sketch::len_hint(&sketch), Some(0));
    }

    #[test]
    #[should_panic(expected = "positive finite")]
    fn non_positive_values_are_rejected() {
        DdSketch::new(0.05).insert_item(&0.0);
    }

    #[test]
    fn estimates_are_exact_for_distinct_single_buckets() {
        let mut sketch = DdSketch::new(0.05);
        sketch.insert_item(&1.0);
        assert_eq!(sketch.count(), 1);
        let median = sketch.median().unwrap();
        // The bucket-mean estimate is within alpha by construction.
        assert!((median / 1.0 - 1.0).abs() < 0.06, "median {median}");
    }

    #[test]
    fn merge_adds_bucket_counts_and_validates_gamma() {
        let mut left = DdSketch::new(0.05);
        let mut right = DdSketch::new(0.05);
        left.insert_item(&1.0);
        left.insert_item(&100.0);
        right.insert_item(&10.0);
        right.insert_item(&100.0);

        left.merge_from(&right).unwrap();
        assert_eq!(left.count(), 4);
        let total: u64 = left.buckets().iter().map(|&(_, c)| c).sum();
        assert_eq!(total, 4);

        let other_gamma = DdSketch::new(0.01);
        assert_eq!(
            left.merge_from(&other_gamma),
            Err(MergeError::GeometryMismatch)
        );
    }

    #[test]
    fn capability_traits_work() {
        let mut sketch = DdSketch::new(0.05);
        Insert::<f64>::insert(&mut sketch, &42.0).unwrap();
        assert_eq!(Sketch::len_hint(&sketch), Some(1));
    }
}
