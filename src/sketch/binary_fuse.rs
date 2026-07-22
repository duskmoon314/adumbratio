//! Binary fuse filter implementation.

use core::hash::{BuildHasher, Hash};

use alloc::boxed::Box;
#[cfg(any(feature = "std", feature = "libm"))]
use alloc::vec::Vec;

use crate::block::Fingerprint;
use crate::hash::{DefaultBuildHasher, mix64, reduce};
use crate::traits::{Contains, Sketch};

/// Maximum construction attempts before giving up. Peeling at the
/// binary-fuse table overhead essentially always succeeds on the first
/// attempt; the boundary-size retry alternates a halved segment length the
/// way the reference implementation does.
#[cfg(any(feature = "std", feature = "libm"))]
const MAX_ATTEMPTS: usize = 100;

/// A static binary fuse filter for approximate membership.
///
/// The successor of the xor filter: each key also maps to three positions,
/// but instead of three equal segments the first position is spread over
/// the whole table and the other two are derived by xor-masking inside
/// power-of-two segments. Peeling then succeeds at a higher load factor,
/// so the table needs about `1.125 * f` bits per item instead of xor's
/// `1.23 * f` (for `f`-bit fingerprints) — the smallest practical
/// membership filter of the family.
///
/// ```text
/// h0 = reduce(hash, SegmentCountLength)
/// h1 = (h0 + SegmentLength) ^ (hash >> 18) & mask
/// h2 = (h1 + SegmentLength) ^ hash & mask
///
/// contains("x"): T[h0] ^ T[h1] ^ T[h2] == fp(x)
/// ```
///
/// Construction and querying follow the reference implementation closely
/// (Graf's FastFilter binaryfusefilter), including its empirical
/// segment-length and size-factor formulas. The trade-offs are the same as
/// the xor filter's: the set is frozen after construction — no insertion,
/// deletion, or merging. The fingerprint type `F` fixes the slot width and
/// the false-positive rate `2^-BITS` (default `u16`, about 1.5e-5).
///
/// # References
///
/// - Thomas Mueller Graf and Daniel Lemire, "Binary Fuse Filters: Fast and
///   Smaller Than Xor Filters", ACM Journal of Experimental Algorithmics,
///   2022. <https://doi.org/10.1145/3510449>
/// - Reference implementation: <https://github.com/FastFilter/xorfilter>
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BinaryFuseFilter<F: Fingerprint = u16, S = DefaultBuildHasher> {
    fingerprints: Box<[F]>,
    segment_length: usize,
    segment_count: usize,
    layout_seed: u64,
    len: usize,
    seed_fingerprint: u64,
    hasher: S,
}

/// Construction parameters ported from the reference implementation. The
/// formulas are empirical and marked sensitive there, so they are kept
/// bit-faithful (including the floor before the shift).
#[cfg(any(feature = "std", feature = "libm"))]
struct Parameters {
    segment_length: usize,
    segment_count: usize,
    fingerprints_len: usize,
}

#[cfg(any(feature = "std", feature = "libm"))]
impl Parameters {
    #[cfg(any(feature = "std", feature = "libm"))]
    fn compute(size: usize) -> Self {
        use crate::float;

        const MAX_SEGMENT_LENGTH: usize = 262_144;

        // 1 << floor(log(size) / log(3.33) + 2.25), capped.
        let raw = float::floor(float::ln(size as f64) / float::ln(3.33) + 2.25);
        let segment_length = (1_usize << raw.max(0.0) as u32).min(MAX_SEGMENT_LENGTH);

        // round(size * max(1.125, 0.875 + 0.25 * log(1e6) / log(size))).
        let capacity = if size > 1 {
            let factor =
                1.125_f64.max(0.875 + 0.25 * float::ln(1_000_000.0) / float::ln(size as f64));
            float::round(size as f64 * factor) as usize
        } else {
            0
        };

        let total_segment_count = capacity
            .div_ceil(segment_length)
            .max(3);
        let segment_count = total_segment_count - 2;
        Self {
            segment_length,
            segment_count,
            fingerprints_len: total_segment_count * segment_length,
        }
    }

    /// The alternate parameters used by every fourth retry on small sets:
    /// halved segment length, doubled-plus-two segment count, same table
    /// footprint (`SegmentCountLength` grows by one segment).
    fn halved(&self) -> Self {
        let segment_length = self.segment_length / 2;
        let segment_count = self.segment_count * 2 + 2;
        Self {
            segment_length,
            segment_count,
            fingerprints_len: self.fingerprints_len,
        }
    }
}

impl BinaryFuseFilter<u16, DefaultBuildHasher> {
    /// Builds a binary fuse filter from `items` with hash seed zero.
    ///
    /// # Panics
    ///
    /// Panics if `items` is empty, or if construction fails after 100
    /// attempts (essentially impossible for the standard table overhead).
    ///
    /// Available with the `std` or `libm` feature.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn build<T: Hash>(items: &[T]) -> Self {
        Self::build_with_seed(items, 0)
    }
}

impl<F: Fingerprint> BinaryFuseFilter<F, DefaultBuildHasher> {
    /// Builds a binary fuse filter from `items` with an explicit hash seed,
    /// using the fingerprint width of `F`. For the default `u16` width,
    /// [`BinaryFuseFilter::build`] is the ergonomic entry point.
    ///
    /// Duplicate items are harmless: construction works on the distinct
    /// 64-bit hashes.
    ///
    /// # Panics
    ///
    /// Panics under the same conditions as [`BinaryFuseFilter::build`].
    ///
    /// Available with the `std` or `libm` feature.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn build_with_seed<T: Hash>(items: &[T], seed: u64) -> Self {
        let hasher = DefaultBuildHasher::new(seed);
        let hashes = distinct_hashes(&hasher, items);
        let built = build_table(&hashes, seed);
        Self {
            fingerprints: built.fingerprints,
            segment_length: built.parameters.segment_length,
            segment_count: built.parameters.segment_count,
            layout_seed: built.layout_seed,
            len: hashes.len(),
            seed_fingerprint: hasher.seed_fingerprint(),
            hasher,
        }
    }
}

impl<F: Fingerprint, S> BinaryFuseFilter<F, S> {
    /// Returns the number of distinct items the filter was built from.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the filter was built from an empty set (always
    /// false; building requires a non-empty input).
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of fingerprint slots in the table.
    pub fn table_len(&self) -> usize {
        self.fingerprints.len()
    }

    /// Returns the segment length (a power of two).
    pub const fn segment_length(&self) -> usize {
        self.segment_length
    }

    /// Returns the seed fingerprint identifying the construction hash.
    pub const fn seed_fingerprint(&self) -> u64 {
        self.seed_fingerprint
    }

    /// Returns the approximate false-positive probability, `2^-BITS`.
    pub fn expected_fpp(&self) -> f64 {
        1.0 / (1_u64 << F::BITS.min(63)) as f64
    }

    /// Returns the byte length of the table storage.
    pub fn storage_bytes(&self) -> usize {
        self.fingerprints.len() * size_of::<F>()
    }

    /// Clears the table.
    pub fn clear(&mut self) {
        self.fingerprints.fill(F::from_u64(0));
        self.len = 0;
    }

    /// The three table positions of a hash, following the reference
    /// implementation's `getHashFromHash`.
    fn positions(&self, hash: u64) -> (usize, usize, usize) {
        let segment_count_length = self.segment_count * self.segment_length;
        let mask = self.segment_length - 1;
        let h0 = reduce(mix64(hash ^ self.layout_seed), segment_count_length);
        let h1 = (h0 + self.segment_length) ^ ((hash >> 18) as usize & mask);
        let h2 = (h1 + self.segment_length) ^ (hash as usize & mask);
        (h0, h1, h2)
    }
}

impl<F: Fingerprint, S> BinaryFuseFilter<F, S>
where
    S: BuildHasher,
{
    /// Returns whether `item` may be present.
    pub fn contains_item<T>(&self, item: &T) -> bool
    where
        T: Hash + ?Sized,
    {
        let hash = crate::hash::hash_one(&self.hasher, item);
        let (h0, h1, h2) = self.positions(hash);
        let fingerprint = F::from_u64(mix64(hash)).to_u64();
        (self.fingerprints[h0].to_u64()
            ^ self.fingerprints[h1].to_u64()
            ^ self.fingerprints[h2].to_u64())
            == fingerprint
    }
}

/// Hashes every item once and drops duplicate hashes, so construction sees
/// a distinct key set.
#[cfg(any(feature = "std", feature = "libm"))]
fn distinct_hashes<S: BuildHasher, T: Hash>(hasher: &S, items: &[T]) -> Vec<u64> {
    let mut hashes: Vec<u64> = items
        .iter()
        .map(|item| crate::hash::hash_one(hasher, item))
        .collect();
    hashes.sort_unstable();
    hashes.dedup();
    hashes
}

#[cfg(any(feature = "std", feature = "libm"))]
struct BuiltTable<F> {
    fingerprints: Box<[F]>,
    parameters: Parameters,
    layout_seed: u64,
}

/// Positions under an alternate parameter set (used for peeling retries).
#[cfg(any(feature = "std", feature = "libm"))]
fn positions_with(hash: u64, parameters: &Parameters, layout_seed: u64) -> (usize, usize, usize) {
    let segment_count_length = parameters.segment_count * parameters.segment_length;
    let mask = parameters.segment_length - 1;
    let h0 = reduce(mix64(hash ^ layout_seed), segment_count_length);
    let h1 = (h0 + parameters.segment_length) ^ ((hash >> 18) as usize & mask);
    let h2 = (h1 + parameters.segment_length) ^ (hash as usize & mask);
    (h0, h1, h2)
}

/// Peels the hypergraph and assigns slot values, retrying with the
/// reference implementation's boundary-size segment-length alternation.
#[cfg(any(feature = "std", feature = "libm"))]
fn build_table<F: Fingerprint>(hashes: &[u64], seed: u64) -> BuiltTable<F> {
    assert!(
        !hashes.is_empty(),
        "cannot build a binary fuse filter from no items"
    );
    let base_parameters = Parameters::compute(hashes.len());
    let small_set = hashes.len() > 4 && hashes.len() < 1_000_000;
    for attempt in 0..MAX_ATTEMPTS {
        // Reference behavior for boundary sizes: every fourth attempt tries
        // the halved segment length, the next restores it.
        let parameters = if small_set && attempt % 4 == 1 {
            base_parameters.halved()
        } else {
            Parameters {
                segment_length: base_parameters.segment_length,
                segment_count: base_parameters.segment_count,
                fingerprints_len: base_parameters.fingerprints_len,
            }
        };
        let layout_seed = mix64(seed.wrapping_add(attempt as u64));
        if let Some(fingerprints) = try_build_once(hashes, &parameters, layout_seed) {
            return BuiltTable {
                fingerprints,
                parameters,
                layout_seed,
            };
        }
    }
    panic!("binary fuse filter construction failed after {MAX_ATTEMPTS} attempts");
}

#[cfg(any(feature = "std", feature = "libm"))]
fn try_build_once<F: Fingerprint>(
    hashes: &[u64],
    parameters: &Parameters,
    layout_seed: u64,
) -> Option<Box<[F]>> {
    let table_len = parameters.fingerprints_len;
    let mut degree = alloc::vec![0_u32; table_len];
    let mut incident = alloc::vec![0_u64; table_len];

    for (index, &hash) in hashes.iter().enumerate() {
        let (h0, h1, h2) = positions_with(hash, parameters, layout_seed);
        for slot in [h0, h1, h2] {
            degree[slot] += 1;
            incident[slot] ^= index as u64;
        }
    }

    // Peel: repeatedly remove the unique remaining item of a degree-1 slot.
    let mut queue: Vec<usize> = (0..table_len).filter(|&slot| degree[slot] == 1).collect();
    let mut assignments: Vec<(u64, usize)> = Vec::with_capacity(hashes.len());
    let mut head = 0;
    while head < queue.len() && assignments.len() < hashes.len() {
        let slot = queue[head];
        head += 1;
        if degree[slot] != 1 {
            continue;
        }
        let index = incident[slot] as usize;
        assignments.push((index as u64, slot));
        let (h0, h1, h2) = positions_with(hashes[index], parameters, layout_seed);
        for other in [h0, h1, h2] {
            degree[other] -= 1;
            incident[other] ^= index as u64;
            if degree[other] == 1 {
                queue.push(other);
            }
        }
    }
    if assignments.len() < hashes.len() {
        return None; // a cycle remains; retry with a different layout
    }

    // Assign in reverse peel order.
    let mut table = alloc::vec![F::from_u64(0); table_len].into_boxed_slice();
    for &(index, slot) in assignments.iter().rev() {
        let hash = hashes[index as usize];
        let mut value = F::from_u64(mix64(hash)).to_u64();
        let (h0, h1, h2) = positions_with(hash, parameters, layout_seed);
        for other in [h0, h1, h2] {
            if other != slot {
                value ^= table[other].to_u64();
            }
        }
        table[slot] = F::from_u64(value);
    }
    Some(table)
}

impl<F: Fingerprint, S> Sketch for BinaryFuseFilter<F, S> {
    fn clear(&mut self) {
        self.clear();
    }

    fn len_hint(&self) -> Option<u64> {
        Some(self.len as u64)
    }

    fn storage_bytes(&self) -> usize {
        self.storage_bytes()
    }
}

impl<T, F: Fingerprint, S> Contains<T> for BinaryFuseFilter<F, S>
where
    T: Hash + ?Sized,
    S: BuildHasher,
{
    fn contains(&self, item: &T) -> bool {
        self.contains_item(item)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::BinaryFuseFilter;
    use crate::traits::{Contains, Sketch};

    #[test]
    fn built_items_have_no_false_negatives() {
        let items: Vec<u64> = (0..10_000).collect();
        let filter = BinaryFuseFilter::build(&items);
        for item in &items {
            assert!(filter.contains_item(item), "missing built item {item}");
        }
        assert_eq!(filter.len(), items.len());
    }

    #[test]
    fn duplicate_items_are_tolerated() {
        let items: Vec<u64> = (0..1_000).chain(0..500).collect();
        let filter = BinaryFuseFilter::build(&items);
        assert_eq!(filter.len(), 1_000);
        for i in 0..1_000_u64 {
            assert!(filter.contains_item(&i));
        }
    }

    #[test]
    fn narrower_tables_work() {
        let items: Vec<u64> = (0..1_000).collect();
        let filter = BinaryFuseFilter::<u8>::build_with_seed(&items, 0);
        for item in &items {
            assert!(filter.contains_item(item));
        }
        assert_eq!(
            filter.storage_bytes(),
            filter.table_len() * size_of::<u8>()
        );
    }

    #[test]
    fn table_is_smaller_than_xor_at_same_width() {
        let items: Vec<u64> = (0..100_000).collect();
        let fuse = BinaryFuseFilter::build(&items);
        let xor = crate::sketch::XorFilter::build(&items);
        assert!(
            fuse.table_len() < xor.table_len(),
            "fuse {} slots should beat xor {} slots",
            fuse.table_len(),
            xor.table_len()
        );
    }

    #[test]
    fn boundary_sizes_construct() {
        // Sizes that exercise the segment-length retry alternation.
        for n in [5_u64, 17, 100, 1_001, 65_537] {
            let items: Vec<u64> = (0..n).collect();
            let filter = BinaryFuseFilter::build(&items);
            for item in &items {
                assert!(filter.contains_item(item), "missing {item} at n = {n}");
            }
        }
    }

    #[test]
    fn clear_empties_the_filter() {
        let items: Vec<u64> = (0..100).collect();
        let mut filter = BinaryFuseFilter::build(&items);
        filter.clear();
        assert_eq!(filter.len(), 0);
        assert!(!filter.contains_item(&42_u64));
    }

    #[test]
    #[should_panic(expected = "cannot build a binary fuse filter from no items")]
    fn empty_input_is_rejected() {
        BinaryFuseFilter::build(&Vec::<u64>::new());
    }

    #[test]
    fn capability_traits_work() {
        let items: Vec<u64> = (0..100).collect();
        let filter = BinaryFuseFilter::build(&items);
        assert!(Contains::<u64>::contains(&filter, &7));
        assert_eq!(Sketch::len_hint(&filter), Some(100));
    }
}
