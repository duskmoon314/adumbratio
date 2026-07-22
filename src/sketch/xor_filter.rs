//! Xor filter implementation.

use core::hash::{BuildHasher, Hash};

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::block::Fingerprint;
use crate::hash::{DefaultBuildHasher, mix64, reduce};
use crate::traits::{Contains, Sketch};

/// Maximum construction attempts before giving up. With the standard 1.23x
/// table overhead the peeling process almost always succeeds on the first
/// attempt, so repeated failure indicates a pathological input rather than
/// bad luck.
const MAX_ATTEMPTS: usize = 100;

/// A static xor filter for approximate membership.
///
/// An xor filter is built once from a fixed set: every key gets three
/// positions in three equal table segments, and construction assigns each
/// table slot so that the xor of a key's three slots equals its
/// fingerprint. Queries then check `T[h0] ^ T[h1] ^ T[h2] == fp` — a small
/// constant number of probes, no cascading writes ever again.
///
/// ```text
/// build(keys):
///   T = [ 0 | 0 | 0 | ... ]   m = 3 * ceil(1.23 * n / 3) slots
///   peel the 3-uniform hypergraph (degree-1 slots first)
///   assign fingerprints in reverse peel order
///
/// contains("x"):
///   T[h0(x)] ^ T[h1(x)] ^ T[h2(x)] == fp(x) ?
/// ```
///
/// The trade against Bloom filters: about `1.23 * log2(1/eps)` bits per
/// item instead of `1.44 * log2(1/eps)`, and typically faster queries — in
/// exchange the set is frozen (no insertion, no deletion, no merge). The
/// fingerprint type `F` fixes both the slot width and the false-positive
/// rate `2^-BITS` (default `u16`, about 1.5e-5).
///
/// # References
///
/// - Thomas Mueller Graf and Daniel Lemire, "Xor Filters: Faster and
///   Smaller Than Bloom and Cuckoo Filters", ACM Journal of Experimental
///   Algorithmics, 2020. <https://doi.org/10.1145/3376122>
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct XorFilter<F: Fingerprint = u16, S = DefaultBuildHasher> {
    table: Box<[F]>,
    block_length: usize,
    layout_seed: u64,
    len: usize,
    seed_fingerprint: u64,
    hasher: S,
}

impl XorFilter<u16, DefaultBuildHasher> {
    /// Builds an xor filter from `items` with hash seed zero.
    ///
    /// # Panics
    ///
    /// Panics if `items` is empty, or if construction fails after 100
    /// attempts (essentially impossible for the standard table overhead).
    pub fn build<T: Hash>(items: &[T]) -> Self {
        Self::build_with_seed(items, 0)
    }
}

impl<F: Fingerprint> XorFilter<F, DefaultBuildHasher> {
    /// Builds an xor filter from `items` with an explicit hash seed, using
    /// the fingerprint width of `F`. For the default `u16` width,
    /// [`XorFilter::build`] is the ergonomic entry point.
    ///
    /// Duplicate items are harmless: construction works on the distinct
    /// 64-bit hashes.
    ///
    /// # Panics
    ///
    /// Panics under the same conditions as [`XorFilter::build`].
    pub fn build_with_seed<T: Hash>(items: &[T], seed: u64) -> Self {
        let hasher = DefaultBuildHasher::new(seed);
        let hashes = distinct_hashes(&hasher, items);
        let built = build_table(&hashes, seed);
        Self {
            table: built.table,
            block_length: built.block_length,
            layout_seed: built.layout_seed,
            len: hashes.len(),
            seed_fingerprint: hasher.seed_fingerprint(),
            hasher,
        }
    }
}

impl<F: Fingerprint, S> XorFilter<F, S> {
    /// Returns the number of distinct items the filter was built from.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the filter was built from an empty set (always
    /// false; building requires a non-empty input).
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of table slots.
    pub fn table_len(&self) -> usize {
        self.table.len()
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
        self.table.len() * size_of::<F>()
    }

    /// Clears the table, turning the filter back into an empty one that
    /// answers `false` everywhere except by coincidence.
    pub fn clear(&mut self) {
        self.table.fill(F::from_u64(0));
        self.len = 0;
    }
}

impl<F: Fingerprint, S> XorFilter<F, S>
where
    S: BuildHasher,
{
    /// Returns whether `item` may be present.
    pub fn contains_item<T>(&self, item: &T) -> bool
    where
        T: Hash + ?Sized,
    {
        let hash = crate::hash::hash_one(&self.hasher, item);
        let (p0, p1, p2) = positions(hash, self.block_length, self.layout_seed);
        let fingerprint = F::from_u64(mix64(hash)).to_u64();
        (self.table[p0].to_u64() ^ self.table[p1].to_u64() ^ self.table[p2].to_u64())
            == fingerprint
    }
}

/// Hashes every item once and drops duplicate hashes, so construction sees
/// a distinct key set.
fn distinct_hashes<S: BuildHasher, T: Hash>(hasher: &S, items: &[T]) -> Vec<u64> {
    let mut hashes: Vec<u64> = items
        .iter()
        .map(|item| crate::hash::hash_one(hasher, item))
        .collect();
    hashes.sort_unstable();
    hashes.dedup();
    hashes
}

/// The three segment positions of a hash: one per equal-sized segment. The
/// layout seed varies between construction attempts and is stored with the
/// filter so queries reproduce the same positions.
fn positions(hash: u64, block_length: usize, layout_seed: u64) -> (usize, usize, usize) {
    let at = |which: usize| {
        let mixed = mix64(hash ^ layout_seed ^ (which as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        which * block_length + reduce(mixed, block_length)
    };
    (at(0), at(1), at(2))
}

/// The result of a successful table construction.
struct BuiltTable<F> {
    table: Box<[F]>,
    block_length: usize,
    layout_seed: u64,
}

/// Peels the 3-uniform hypergraph and assigns slot values, retrying with a
/// re-seeded layout when the graph has a cycle.
fn build_table<F: Fingerprint>(hashes: &[u64], seed: u64) -> BuiltTable<F> {
    assert!(!hashes.is_empty(), "cannot build an xor filter from no items");
    for attempt in 0..MAX_ATTEMPTS {
        let layout_seed = mix64(seed.wrapping_add(attempt as u64));
        if let Some(built) = try_build_once(hashes, layout_seed) {
            return built;
        }
    }
    panic!("xor filter construction failed after {MAX_ATTEMPTS} attempts");
}

fn try_build_once<F: Fingerprint>(hashes: &[u64], layout_seed: u64) -> Option<BuiltTable<F>> {
    // ceil(1.23 * n / 3) in exact integer arithmetic (no float math needed).
    let block_length = (123 * hashes.len() as u128).div_ceil(300) as usize;
    let table_len = 3 * block_length;

    // Degree counting plus an xor-accumulator of incident item indices:
    // when a slot's degree drops to one, the accumulator names the item.
    let mut degree = alloc::vec![0_u32; table_len];
    let mut incident = alloc::vec![0_u64; table_len];

    for (index, &hash) in hashes.iter().enumerate() {
        let (p0, p1, p2) = positions(hash, block_length, layout_seed);
        for slot in [p0, p1, p2] {
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
        let (p0, p1, p2) = positions(hashes[index], block_length, layout_seed);
        for other in [p0, p1, p2] {
            degree[other] -= 1;
            incident[other] ^= index as u64;
            if degree[other] == 1 {
                queue.push(other);
            }
        }
    }
    if assignments.len() < hashes.len() {
        return None; // a cycle remains; retry with a different layout seed
    }

    // Assign in reverse peel order: when a slot is filled, the other two
    // slots of its item are already final.
    let mut table = alloc::vec![F::from_u64(0); table_len].into_boxed_slice();
    for &(index, slot) in assignments.iter().rev() {
        let hash = hashes[index as usize];
        let mut value = F::from_u64(mix64(hash)).to_u64();
        let (p0, p1, p2) = positions(hash, block_length, layout_seed);
        for other in [p0, p1, p2] {
            if other != slot {
                value ^= table[other].to_u64();
            }
        }
        table[slot] = F::from_u64(value);
    }
    Some(BuiltTable {
        table,
        block_length,
        layout_seed,
    })
}

impl<F: Fingerprint, S> Sketch for XorFilter<F, S> {
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

impl<T, F: Fingerprint, S> Contains<T> for XorFilter<F, S>
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

    use super::XorFilter;
    use crate::traits::{Contains, Sketch};

    #[test]
    fn built_items_have_no_false_negatives() {
        let items: Vec<u64> = (0..10_000).collect();
        let filter = XorFilter::build(&items);
        for item in &items {
            assert!(filter.contains_item(item), "missing built item {item}");
        }
        assert_eq!(filter.len(), items.len());
    }

    #[test]
    fn duplicate_items_are_tolerated() {
        let items: Vec<u64> = (0..1_000).chain(0..500).collect();
        let filter = XorFilter::build(&items);
        assert_eq!(filter.len(), 1_000);
        for i in 0..1_000_u64 {
            assert!(filter.contains_item(&i));
        }
    }

    #[test]
    fn narrower_tables_work() {
        let items: Vec<u64> = (0..1_000).collect();
        let filter = XorFilter::<u8>::build_with_seed(&items, 0);
        for item in &items {
            assert!(filter.contains_item(item));
        }
        assert_eq!(
            filter.storage_bytes(),
            filter.table_len() * size_of::<u8>()
        );
    }

    #[test]
    fn clear_empties_the_filter() {
        let items: Vec<u64> = (0..100).collect();
        let mut filter = XorFilter::build(&items);
        filter.clear();
        assert_eq!(filter.len(), 0);
        assert_eq!(Sketch::len_hint(&filter), Some(0));
        assert!(!filter.contains_item(&42_u64));
    }

    #[test]
    #[should_panic(expected = "cannot build an xor filter from no items")]
    fn empty_input_is_rejected() {
        XorFilter::build(&Vec::<u64>::new());
    }

    #[test]
    fn capability_traits_work() {
        let items: Vec<u64> = (0..100).collect();
        let filter = XorFilter::build(&items);
        assert!(Contains::<u64>::contains(&filter, &7));
        assert_eq!(Sketch::len_hint(&filter), Some(100));
    }
}
