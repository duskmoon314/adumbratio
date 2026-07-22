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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

/// Enhanced double hashing with a cubic perturbation term.
///
/// Extends the Kirsch-Mitzenmacher sequence with an `(i^3 - i) / 6` term,
/// which removes the small-range artifacts of plain double hashing at
/// negligible cost.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EnhancedDoubleHashing;

impl IndexScheme for EnhancedDoubleHashing {
    fn indices(&self, hash: u64, k: usize, m: usize) -> impl Iterator<Item = usize> {
        assert!(m > 0, "index range must be non-zero");
        EnhancedDoubleHashIter {
            h1: mix64(hash),
            h2: mix64(hash ^ 0xa076_1d64_78bd_642f) | 1,
            next: 0,
            k,
            m,
        }
    }
}

#[derive(Clone, Debug)]
struct EnhancedDoubleHashIter {
    h1: u64,
    h2: u64,
    next: usize,
    k: usize,
    m: usize,
}

impl Iterator for EnhancedDoubleHashIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.k {
            return None;
        }

        let i = self.next as u64;
        self.next += 1;
        let perturb = i.wrapping_mul(i.wrapping_mul(i).wrapping_sub(1)) / 6;
        let hash = self
            .h1
            .wrapping_add(i.wrapping_mul(self.h2))
            .wrapping_add(perturb);
        Some(reduce(hash, self.m))
    }
}

/// Partitioned indexing: index `i` falls in partition `[i*m/k, (i+1)*m/k)`.
///
/// Each of the `k` positions is drawn from its own contiguous slice of the
/// range, the scheme behind partitioned Bloom filters. (The Count-Min
/// family uses the same idea row-wise; see [`row_index`].)
///
/// # Panics
///
/// Panics if `m` is zero or smaller than `k`, since every partition must be
/// non-empty.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Partitioned;

impl IndexScheme for Partitioned {
    fn indices(&self, hash: u64, k: usize, m: usize) -> impl Iterator<Item = usize> {
        assert!(m > 0, "index range must be non-zero");
        assert!(
            m >= k,
            "partitioned indexing needs at least as many positions as indices"
        );
        PartitionedIter {
            hash,
            next: 0,
            k,
            m,
        }
    }
}

#[derive(Clone, Debug)]
struct PartitionedIter {
    hash: u64,
    next: usize,
    k: usize,
    m: usize,
}

impl Iterator for PartitionedIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.k {
            return None;
        }

        let i = self.next;
        self.next += 1;
        let start = i * self.m / self.k;
        let end = (i + 1) * self.m / self.k;
        let slice_hash = mix64(self.hash ^ (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        Some(start + reduce(slice_hash, end - start))
    }
}

/// Blocked indexing for cache-friendly filters.
///
/// The first derived position selects a contiguous block of `block_bits`
/// cells (512 bits, one typical cache line, by default); all `k` indices
/// then land inside that block. A lookup touches a single cache line
/// instead of `k` scattered ones, at the cost of a slightly higher
/// false-positive rate.
///
/// If the range length is not a multiple of `block_bits`, the last block is
/// smaller and its indices are reduced into the available tail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Blocked {
    block_bits: usize,
}

impl Blocked {
    /// The default block size: 512 cells, one 64-byte cache line of bits.
    pub const CACHE_LINE_BITS: usize = 512;

    /// Creates a blocked scheme with `block_bits` cells per block.
    ///
    /// # Panics
    ///
    /// Panics if `block_bits` is zero.
    pub const fn new(block_bits: usize) -> Self {
        assert!(block_bits > 0, "block size must be non-zero");
        Self { block_bits }
    }

    /// Returns the configured block size in cells.
    pub const fn block_bits(&self) -> usize {
        self.block_bits
    }
}

impl Default for Blocked {
    fn default() -> Self {
        Self::new(Self::CACHE_LINE_BITS)
    }
}

impl IndexScheme for Blocked {
    fn indices(&self, hash: u64, k: usize, m: usize) -> impl Iterator<Item = usize> {
        assert!(m > 0, "index range must be non-zero");
        let blocks = m.div_ceil(self.block_bits);
        let block = reduce(mix64(hash ^ 0x243f_6a88_85a3_08d3), blocks);
        let start = block * self.block_bits;
        let size = (m - start).min(self.block_bits);
        BlockedIter {
            hash,
            start,
            size,
            next: 0,
            k,
        }
    }
}

#[derive(Clone, Debug)]
struct BlockedIter {
    hash: u64,
    start: usize,
    size: usize,
    next: usize,
    k: usize,
}

impl Iterator for BlockedIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.k {
            return None;
        }

        let i = self.next as u64;
        self.next += 1;
        let lane = mix64(
            self.hash ^ i.wrapping_add(1).wrapping_mul(0x9e37_79b9_7f4a_7c15),
        );
        Some(self.start + reduce(lane, self.size))
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
    use alloc::vec::Vec;

    use super::{
        Blocked, DoubleHashing, EnhancedDoubleHashing, IndexScheme, Partitioned, reduce, row_index,
        sign,
    };

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
    fn enhanced_double_hashing_yields_in_range_indices() {
        for hash in [0, 1, 42, u64::MAX] {
            let indices: Vec<_> = EnhancedDoubleHashing.indices(hash, 9, 1_000).collect();
            assert_eq!(indices.len(), 9);
            assert!(indices.iter().all(|&i| i < 1_000));
        }
    }

    #[test]
    fn partitioned_indices_land_in_their_own_slices() {
        let (k, m) = (7, 1_001);
        for (i, index) in Partitioned.indices(123, k, m).enumerate() {
            let start = i * m / k;
            let end = (i + 1) * m / k;
            assert!((start..end).contains(&index), "index {index} outside slice {start}..{end}");
        }
    }

    #[test]
    fn partitioned_covers_whole_range() {
        // With k == m every partition is a single cell, so the indices are a
        // permutation of 0..m.
        let m = 64;
        let mut indices: Vec<_> = Partitioned.indices(42, m, m).collect();
        indices.sort_unstable();
        assert_eq!(indices, (0..m).collect::<Vec<_>>());
    }

    #[test]
    fn blocked_indices_stay_inside_one_block() {
        let scheme = Blocked::new(64);
        let m = 1_000; // deliberately not a multiple of the block size
        for hash in [0, 1, 42, u64::MAX / 3, u64::MAX] {
            let indices: Vec<_> = scheme.indices(hash, 8, m).collect();
            assert_eq!(indices.len(), 8);
            assert!(indices.iter().all(|&i| i < m));
            let block = indices[0] / 64;
            assert!(
                indices.iter().all(|&i| i / 64 == block),
                "indices {indices:?} span more than one block"
            );
        }
    }

    #[test]
    fn blocked_handles_single_partial_block() {
        let scheme = Blocked::new(512);
        let indices: Vec<_> = scheme.indices(7, 5, 100).collect();
        assert_eq!(indices.len(), 5);
        assert!(indices.iter().all(|&i| i < 100));
    }

    #[test]
    fn row_indices_and_signs_are_in_range() {
        for row in 0..10 {
            assert!(row_index(123, row, 17) < 17);
            assert!(matches!(sign(123, row), -1 | 1));
        }
    }
}
