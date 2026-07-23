//! Entropy sampler: Shannon entropy estimation from frequency samples.

use core::hash::{BuildHasher, Hash};

use alloc::boxed::Box;
use alloc::vec;

use crate::error::MergeError;
use crate::hash::{DefaultBuildHasher, hash_one, mix64};
use crate::traits::{Insert, Merge, Sketch};

/// An entropy sampler: `k` independent uniform samples of the event
/// stream, from which Shannon entropy is estimated given any frequency
/// oracle.
///
/// The estimator is the hashing-based one from the priority-sampling
/// literature: each of `k` slots keeps the event with the largest derived
/// value, which makes every slot a *uniform sample over occurrences*. A
/// uniform sample of occurrences is an item drawn with probability
/// proportional to its frequency, so
///
/// ```text
/// H = -sum_x p(x) log p(x)  =  E[ -log p(sample) ]
///
/// estimate = mean over slots of -log2(f(slot) / N)
/// ```
///
/// where `f(slot)` is the sampled item's frequency. Each slot is exact;
/// the only approximation is the frequency oracle, which can be exact (a
/// counter map) or a frequency sketch from this crate
/// ([`CountMinSketch::estimate_hash`](crate::sketch::CountMinSketch::estimate_hash)
/// composes directly). Averaging `k` slots scales the estimator variance
/// by `1/k` — `k ≈ 1/eps²` for relative error `eps`.
///
/// Merging is exact too: the largest derived value over the union is the
/// larger of the two tables' values, per slot.
///
/// # References
///
/// - Nick Duffield, Mikkel Thorup, and Carsten Lund, "Priority Sampling
///   for Estimating Arbitrary Subset Sums", Journal of the ACM, 2007.
///   <https://doi.org/10.1145/1314690.1314696>
/// - Zaoxing Liu, Antonis Manousis, Gregory Vorsanger, Vyas Sekar, and
///   Vladimir Braverman, "One Sketch to Rule Them All: Rethinking Network
///   Flow Monitoring with UnivMon", SIGCOMM 2016.
///   <https://doi.org/10.1145/2934872.2934908>
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EntropySampler<S = DefaultBuildHasher> {
    best: Box<[u64]>,
    keys: Box<[u64]>,
    total: u64,
    seed_fingerprint: u64,
    hasher: S,
}

impl EntropySampler<DefaultBuildHasher> {
    /// Creates an entropy sampler with `samples` slots and seed zero.
    ///
    /// # Panics
    ///
    /// Panics if `samples` is zero.
    pub fn new(samples: usize) -> Self {
        Self::with_seed(samples, 0)
    }

    /// Creates an entropy sampler with an explicit hash seed.
    ///
    /// # Panics
    ///
    /// Panics if `samples` is zero.
    pub fn with_seed(samples: usize, seed: u64) -> Self {
        let hasher = DefaultBuildHasher::new(seed);
        Self::from_parts(samples, hasher.seed_fingerprint(), hasher)
    }
}

impl<S> EntropySampler<S> {
    /// Creates an entropy sampler from explicit components.
    ///
    /// # Panics
    ///
    /// Panics if `samples` is zero.
    pub fn from_parts(samples: usize, seed_fingerprint: u64, hasher: S) -> Self {
        assert!(samples > 0, "entropy sampler needs at least one slot");
        Self {
            best: vec![0; samples].into_boxed_slice(),
            keys: vec![0; samples].into_boxed_slice(),
            total: 0,
            seed_fingerprint,
            hasher,
        }
    }

    /// Returns the number of sample slots.
    pub fn samples(&self) -> usize {
        self.best.len()
    }

    /// Returns the total number of inserted events.
    pub const fn total_count(&self) -> u64 {
        self.total
    }

    /// Returns the seed fingerprint used by merge compatibility checks.
    pub const fn seed_fingerprint(&self) -> u64 {
        self.seed_fingerprint
    }

    /// Returns the item hashes held by the non-empty slots.
    pub fn sampled_hashes(&self) -> impl Iterator<Item = u64> + '_ {
        self.keys.iter().copied().filter(|&key| key != 0)
    }

    /// Returns the byte length of the slot storage.
    pub fn storage_bytes(&self) -> usize {
        2 * self.best.len() * size_of::<u64>()
    }

    /// Clears all slots and the event count.
    pub fn clear(&mut self) {
        self.best.fill(0);
        self.keys.fill(0);
        self.total = 0;
    }
}

impl<S> EntropySampler<S>
where
    S: BuildHasher,
{
    /// Inserts `item`, replacing a slot whenever the event's derived value
    /// is larger than the stored one.
    pub fn insert_item<T>(&mut self, item: &T)
    where
        T: Hash + ?Sized,
    {
        let hash = hash_one(&self.hasher, item);
        let event = self.total;
        self.total += 1;
        for (index, (slot, key)) in self.best.iter_mut().zip(self.keys.iter_mut()).enumerate() {
            let value = mix64(
                self.seed_fingerprint ^ event ^ (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
            );
            if value > *slot {
                *slot = value;
                *key = hash;
            }
        }
    }

    /// Estimates Shannon entropy in bits: the mean of `-log2(f / N)` over
    /// non-empty slots, where `f` comes from the supplied frequency oracle
    /// mapping item hashes to counts.
    ///
    /// With an exact oracle the estimator is unbiased; with a frequency
    /// sketch its error is the oracle's error passed through, plus the
    /// `O(1/sqrt(k))` sampling noise.
    ///
    /// Available with the `std` or `libm` feature.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn shannon_entropy(&self, estimate: impl Fn(u64) -> u64) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let mut sum = 0.0_f64;
        let mut occupied = 0_usize;
        for key in self.sampled_hashes() {
            let frequency = estimate(key);
            if frequency > 0 {
                let p = frequency as f64 / self.total as f64;
                sum += -crate::float::log2(p);
                occupied += 1;
            }
        }
        if occupied == 0 {
            0.0
        } else {
            sum / occupied as f64
        }
    }
}

impl<S> Sketch for EntropySampler<S> {
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

impl<T, S> Insert<T> for EntropySampler<S>
where
    T: Hash + ?Sized,
    S: BuildHasher,
{
    type Err = core::convert::Infallible;

    fn insert(&mut self, item: &T) -> Result<(), Self::Err> {
        self.insert_item(item);
        Ok(())
    }
}

impl<S> Merge for EntropySampler<S> {
    /// Merges by taking the larger derived value per slot and adding event
    /// counts. Requires equal slot counts and seeds.
    ///
    /// Derived values are table-local (they use each table's own event
    /// counter), so the merged sample is *not* identical to a from-scratch
    /// sample of the combined stream — but it is still a uniform sample
    /// over the combined events, hence an equally valid entropy estimator.
    fn merge_from(&mut self, other: &Self) -> Result<(), MergeError> {
        if self.best.len() != other.best.len() {
            return Err(MergeError::GeometryMismatch);
        }
        if self.seed_fingerprint != other.seed_fingerprint {
            return Err(MergeError::SeedMismatch);
        }
        for index in 0..self.best.len() {
            if other.best[index] > self.best[index] {
                self.best[index] = other.best[index];
                self.keys[index] = other.keys[index];
            }
        }
        self.total = self.total.saturating_add(other.total);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::EntropySampler;
    use crate::error::MergeError;
    use crate::traits::{Insert, Merge, Sketch};

    #[test]
    fn empty_sampler_has_zero_entropy() {
        let sampler = EntropySampler::new(16);
        assert_eq!(sampler.shannon_entropy(|_| 1), 0.0);
        assert_eq!(sampler.sampled_hashes().count(), 0);
    }

    #[test]
    fn single_outcome_stream_has_zero_entropy() {
        let mut sampler = EntropySampler::new(16);
        for _ in 0..100 {
            sampler.insert_item(&"same");
        }
        assert_eq!(sampler.total_count(), 100);
        // Every slot samples the only item with p = 1: H = 0.
        assert_eq!(sampler.shannon_entropy(|_| 100), 0.0);
    }

    #[test]
    fn merge_yields_a_valid_sample_and_validates() {
        let mut left = EntropySampler::with_seed(256, 1);
        let mut right = EntropySampler::with_seed(256, 1);
        let mut whole = EntropySampler::with_seed(256, 1);
        let mut counts = alloc::collections::BTreeMap::new();
        for i in 0..2_000_u64 {
            left.insert_item(&i);
            whole.insert_item(&i);
            *counts
                .entry(crate::hash::hash_one(&crate::hash::DefaultBuildHasher::new(1), &i))
                .or_insert(0) += 1;
        }
        for i in 2_000..4_000_u64 {
            right.insert_item(&i);
            whole.insert_item(&i);
            *counts
                .entry(crate::hash::hash_one(&crate::hash::DefaultBuildHasher::new(1), &i))
                .or_insert(0) += 1;
        }

        left.merge_from(&right).unwrap();
        assert_eq!(left.total_count(), 4_000);

        // The merged sample is not identical to a from-scratch one, but
        // both are uniform over the combined events, so their entropy
        // estimates agree within sampling noise.
        let oracle = |hash: u64| *counts.get(&hash).unwrap_or(&0);
        let merged_estimate = left.shannon_entropy(oracle);
        let direct_estimate = whole.shannon_entropy(oracle);
        assert!(
            (merged_estimate - direct_estimate).abs() <= 0.2,
            "merged {merged_estimate} vs direct {direct_estimate}"
        );

        let other_seed = EntropySampler::with_seed(256, 2);
        assert_eq!(left.merge_from(&other_seed), Err(MergeError::SeedMismatch));
    }

    #[test]
    fn capability_traits_work() {
        let mut sampler = EntropySampler::new(8);
        Insert::<str>::insert(&mut sampler, "alice").unwrap();
        assert_eq!(Sketch::len_hint(&sampler), Some(1));
    }
}
