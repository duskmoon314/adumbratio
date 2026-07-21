//! Fingerprint derivation for Cuckoo-style filters.

use crate::block::Fingerprint;

use super::{mix64, reduce};

/// Partial-key Cuckoo hashing for filters.
///
/// The alternate bucket is derived from the current bucket and fingerprint,
/// which makes the mapping an involution: applying it twice returns the
/// original bucket.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PartialKeyCuckoo;

impl PartialKeyCuckoo {
    /// Derives a non-empty fingerprint with at most `bits` significant bits.
    ///
    /// # Panics
    ///
    /// Panics if `bits` is not in `1..=63`.
    pub fn fingerprint(bits: u32, hash: u64) -> u64 {
        assert!(
            (1..=63).contains(&bits),
            "fingerprint width must be in 1..=63 bits"
        );
        let mask = (1_u64 << bits) - 1;
        let fp = mix64(hash) & mask;
        if fp == 0 { 1 } else { fp }
    }

    /// Derives a typed non-empty fingerprint from `hash`.
    pub fn typed_fingerprint<F: Fingerprint>(hash: u64) -> F {
        F::from_hash(mix64(hash))
    }

    /// Returns the primary bucket for `hash`.
    ///
    /// # Panics
    ///
    /// Panics if `buckets` is zero.
    pub fn bucket(hash: u64, buckets: usize) -> usize {
        reduce(mix64(hash ^ 0x8ebc_6af0_9c88_c6e3), buckets)
    }

    /// Returns the alternate bucket for `bucket` and `fingerprint`.
    ///
    /// `buckets` must be a power of two for the standard XOR mapping. The
    /// returned bucket satisfies `alt_bucket(alt_bucket(b, fp), fp) == b`.
    ///
    /// # Panics
    ///
    /// Panics if `buckets` is zero or not a power of two, or if `bucket` is out
    /// of range.
    pub fn alt_bucket(bucket: usize, fingerprint: u64, buckets: usize) -> usize {
        assert!(buckets > 0, "bucket count must be greater than zero");
        assert!(
            buckets.is_power_of_two(),
            "partial-key cuckoo bucket count must be a power of two"
        );
        assert!(
            bucket < buckets,
            "bucket index {bucket} out of bounds for {buckets} buckets"
        );
        if buckets == 1 {
            return 0;
        }
        let offset = reduce(mix64(fingerprint), buckets - 1) + 1;
        (bucket ^ offset) & (buckets - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::PartialKeyCuckoo;

    #[test]
    fn alternate_bucket_is_an_involution() {
        for buckets in [2, 4, 16, 256] {
            for bucket in 0..buckets {
                for fp in [1, 2, 17, 999] {
                    let alt = PartialKeyCuckoo::alt_bucket(bucket, fp, buckets);
                    assert_eq!(PartialKeyCuckoo::alt_bucket(alt, fp, buckets), bucket);
                }
            }
        }
    }
}
