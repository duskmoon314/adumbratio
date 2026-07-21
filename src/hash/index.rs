//! Index derivation schemes.

use super::mix64;

/// A strategy for deriving storage indices from one item hash.
pub trait IndexScheme {
    /// Returns `k` indices in the range `0..m`.
    ///
    /// # Panics
    ///
    /// Implementations panic if `m` is zero.
    fn indices(&self, hash: u64, k: usize, m: usize) -> impl Iterator<Item = usize>;
}

/// Kirsch-Mitzenmacher double hashing.
///
/// The scheme derives two mixed hashes from one item hash and emits
/// `h1 + i * h2`, reduced into the target range. It is the default index
/// scheme for Bloom-style filters in this crate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DoubleHashing;

impl IndexScheme for DoubleHashing {
    fn indices(&self, hash: u64, k: usize, m: usize) -> impl Iterator<Item = usize> {
        assert!(m > 0, "index range must be non-zero");
        DoubleHashIter {
            h1: mix64(hash),
            h2: mix64(hash ^ 0xa076_1d64_78bd_642f) | 1,
            next: 0,
            k,
            m,
        }
    }
}

#[derive(Clone, Debug)]
struct DoubleHashIter {
    h1: u64,
    h2: u64,
    next: usize,
    k: usize,
    m: usize,
}

impl Iterator for DoubleHashIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.k {
            return None;
        }

        let step = self.next as u64;
        self.next += 1;
        let hash = self.h1.wrapping_add(step.wrapping_mul(self.h2));
        Some(reduce(hash, self.m))
    }
}

/// Reduces a 64-bit hash into `0..range` with multiply-shift reduction.
///
/// # Panics
///
/// Panics if `range` is zero.
pub fn reduce(hash: u64, range: usize) -> usize {
    assert!(range > 0, "reduction range must be non-zero");
    ((hash as u128 * range as u128) >> u64::BITS) as usize
}

/// Derives a row-specific index in the range `0..width`.
///
/// This helper is used by matrix sketches such as Count-Min and Count Sketch
/// to get one independent-looking column per row from a single item hash.
///
/// # Panics
///
/// Panics if `width` is zero.
pub fn row_index(hash: u64, row: usize, width: usize) -> usize {
    let row_hash = mix64(hash ^ (row as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    reduce(row_hash, width)
}

/// Derives a Count Sketch sign from a hash and row.
///
/// The returned value is always either `-1` or `1`.
pub fn sign(hash: u64, row: usize) -> i64 {
    let sign_hash = mix64(hash ^ (row as u64).wrapping_mul(0xd1b5_4a32_d192_ed03));
    if sign_hash & 1 == 0 { 1 } else { -1 }
}

#[cfg(test)]
mod tests {
    use super::{DoubleHashing, IndexScheme, reduce, row_index, sign};

    #[test]
    fn reduction_stays_in_range() {
        for range in [1, 2, 3, 17, 1_000, 1 << 20] {
            for hash in [0, 1, 42, u64::MAX / 2, u64::MAX] {
                assert!(reduce(hash, range) < range);
            }
        }
    }

    #[test]
    fn double_hashing_yields_requested_number_of_indices() {
        let indices: Vec<_> = DoubleHashing.indices(123, 7, 128).collect();
        assert_eq!(indices.len(), 7);
        assert!(indices.iter().all(|&i| i < 128));
    }

    #[test]
    fn row_indices_and_signs_are_in_range() {
        for row in 0..10 {
            assert!(row_index(123, row, 17) < 17);
            assert!(matches!(sign(123, row), -1 | 1));
        }
    }
}
