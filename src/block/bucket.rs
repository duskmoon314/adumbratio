use core::fmt::Debug;

use alloc::boxed::Box;
use alloc::vec;

use crate::error::BucketFull;
use crate::policy::RngLite;

/// A fingerprint value that reserves zero as the empty slot marker.
pub trait Fingerprint: Copy + Debug + Eq {
    /// The empty-slot marker.
    const EMPTY: Self;

    /// The number of bits stored by this fingerprint type.
    const BITS: u32;

    /// Builds a fingerprint from a non-zero integer.
    fn from_nonzero_u64(value: u64) -> Self;

    /// Builds a fingerprint from an integer, truncating excess bits. Unlike
    /// [`Fingerprint::from_hash`], zero is preserved — for structures like
    /// xor-filter tables where slots have no vacancy marker.
    fn from_u64(value: u64) -> Self;

    /// Converts the fingerprint to an integer.
    fn to_u64(self) -> u64;

    /// Builds a non-empty fingerprint from a hash value.
    fn from_hash(hash: u64) -> Self {
        let mask = if Self::BITS == u64::BITS {
            u64::MAX
        } else {
            (1_u64 << Self::BITS) - 1
        };
        let mut value = hash & mask;
        if value == 0 {
            value = 1;
        }
        Self::from_nonzero_u64(value)
    }

    /// Returns whether this fingerprint is the empty marker.
    fn is_empty(self) -> bool {
        self == Self::EMPTY
    }
}

macro_rules! impl_fingerprint {
    ($ty:ty) => {
        impl Fingerprint for $ty {
            const EMPTY: Self = 0;
            const BITS: u32 = <$ty>::BITS;

            fn from_nonzero_u64(value: u64) -> Self {
                debug_assert_ne!(value, 0);
                value as Self
            }

            fn from_u64(value: u64) -> Self {
                value as Self
            }

            fn to_u64(self) -> u64 {
                self as u64
            }
        }
    };
}

impl_fingerprint!(u8);
impl_fingerprint!(u16);
impl_fingerprint!(u32);
impl_fingerprint!(u64);

/// Fixed-size buckets of fingerprint slots.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BucketArray<F: Fingerprint> {
    slots: Box<[F]>,
    buckets: usize,
    slots_per_bucket: usize,
    occupancy: usize,
}

impl<F: Fingerprint> BucketArray<F> {
    /// Creates `buckets` buckets, each containing `slots_per_bucket` slots.
    ///
    /// # Panics
    ///
    /// Panics if either dimension is zero or if their product overflows.
    pub fn new(buckets: usize, slots_per_bucket: usize) -> Self {
        assert!(buckets > 0, "bucket count must be greater than zero");
        assert!(
            slots_per_bucket > 0,
            "slots per bucket must be greater than zero"
        );
        let len = buckets
            .checked_mul(slots_per_bucket)
            .expect("bucket array length overflowed usize");
        Self {
            slots: vec![F::EMPTY; len].into_boxed_slice(),
            buckets,
            slots_per_bucket,
            occupancy: 0,
        }
    }

    /// Returns the number of buckets.
    pub const fn buckets(&self) -> usize {
        self.buckets
    }

    /// Returns the number of slots in each bucket.
    pub const fn slots_per_bucket(&self) -> usize {
        self.slots_per_bucket
    }

    /// Returns the number of occupied slots.
    pub const fn occupancy(&self) -> usize {
        self.occupancy
    }

    /// Returns whether `bucket` contains `fp`.
    ///
    /// Empty fingerprints never match.
    ///
    /// # Panics
    ///
    /// Panics if `bucket` is out of bounds.
    pub fn contains(&self, bucket: usize, fp: F) -> bool {
        self.bucket_slots(bucket)
            .iter()
            .any(|&slot| !fp.is_empty() && slot == fp)
    }

    /// Inserts `fp` into the first empty slot in `bucket`.
    ///
    /// # Errors
    ///
    /// Returns [`BucketFull`] if the bucket has no empty slot.
    ///
    /// # Panics
    ///
    /// Panics if `bucket` is out of bounds or if `fp` is empty.
    pub fn try_insert(&mut self, bucket: usize, fp: F) -> Result<(), BucketFull> {
        assert!(!fp.is_empty(), "cannot insert empty fingerprint");
        for slot in self.bucket_slots_mut(bucket) {
            if slot.is_empty() {
                *slot = fp;
                self.occupancy += 1;
                return Ok(());
            }
        }
        Err(BucketFull)
    }

    /// Removes `fp` from `bucket`, returning whether a slot was cleared.
    ///
    /// # Panics
    ///
    /// Panics if `bucket` is out of bounds.
    pub fn remove(&mut self, bucket: usize, fp: F) -> bool {
        for slot in self.bucket_slots_mut(bucket) {
            if !fp.is_empty() && *slot == fp {
                *slot = F::EMPTY;
                self.occupancy -= 1;
                return true;
            }
        }
        false
    }

    /// Replaces a random slot in `bucket` with `fp` and returns the evicted value.
    ///
    /// # Panics
    ///
    /// Panics if `bucket` is out of bounds or if `fp` is empty.
    pub fn swap_random_slot(&mut self, bucket: usize, fp: F, rng: &mut impl RngLite) -> F {
        assert!(!fp.is_empty(), "cannot insert empty fingerprint");
        self.check_bucket(bucket);
        let slot = rng.next_index(self.slots_per_bucket);
        let index = self.offset(bucket) + slot;
        let evicted = self.slots[index];
        self.slots[index] = fp;
        evicted
    }

    /// Clears all buckets.
    pub fn clear(&mut self) {
        self.slots.fill(F::EMPTY);
        self.occupancy = 0;
    }

    /// Returns the byte length of the backing slot storage.
    pub fn storage_bytes(&self) -> usize {
        self.slots.len() * size_of::<F>()
    }

    /// Iterates over occupied `(bucket, fingerprint)` pairs.
    pub fn occupied(&self) -> impl Iterator<Item = (usize, F)> + '_ {
        self.slots
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, fp)| !fp.is_empty())
            .map(|(index, fp)| (index / self.slots_per_bucket, fp))
    }

    fn bucket_slots(&self, bucket: usize) -> &[F] {
        self.check_bucket(bucket);
        let start = self.offset(bucket);
        &self.slots[start..start + self.slots_per_bucket]
    }

    fn bucket_slots_mut(&mut self, bucket: usize) -> &mut [F] {
        self.check_bucket(bucket);
        let start = self.offset(bucket);
        &mut self.slots[start..start + self.slots_per_bucket]
    }

    fn offset(&self, bucket: usize) -> usize {
        bucket * self.slots_per_bucket
    }

    fn check_bucket(&self, bucket: usize) {
        assert!(
            bucket < self.buckets,
            "bucket index {bucket} out of bounds for {} buckets",
            self.buckets
        );
    }
}

#[cfg(test)]
mod tests {
    use super::BucketArray;
    use crate::policy::XorShift64;

    #[test]
    fn insert_contains_remove_and_clear() {
        let mut buckets = BucketArray::<u8>::new(2, 2);
        buckets.try_insert(0, 7).unwrap();

        assert!(buckets.contains(0, 7));
        assert_eq!(buckets.occupancy(), 1);
        assert!(buckets.remove(0, 7));
        assert!(!buckets.contains(0, 7));
        assert_eq!(buckets.occupancy(), 0);

        buckets.try_insert(1, 9).unwrap();
        buckets.clear();
        assert_eq!(buckets.occupancy(), 0);
    }

    #[test]
    fn full_bucket_returns_error_and_swap_evicts() {
        let mut buckets = BucketArray::<u16>::new(1, 2);
        let mut rng = XorShift64::new(1);
        buckets.try_insert(0, 10).unwrap();
        buckets.try_insert(0, 11).unwrap();

        assert!(buckets.try_insert(0, 12).is_err());
        let evicted = buckets.swap_random_slot(0, 12, &mut rng);
        assert!(evicted == 10 || evicted == 11);
        assert!(buckets.contains(0, 12));
        assert_eq!(buckets.occupancy(), 2);
    }
}
