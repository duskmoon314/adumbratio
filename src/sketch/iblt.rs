//! IBLT: invertible Bloom lookup table for set reconciliation.

use core::hash::{BuildHasher, Hash};

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use crate::error::{DecodeError, MergeError};
use crate::hash::{DefaultBuildHasher, hash_one, mix64, reduce};
use crate::traits::{Contains, Insert, Merge, Remove, Sketch};

/// One IBLT cell: an occurrence count, the xor of keys, and the xor of
/// their verification hashes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct IbltCell {
    count: i32,
    key_sum: u64,
    hash_sum: u64,
}

/// The result of reconciling two sketches: the hashes present in exactly
/// one of them, split by side.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Reconciliation {
    /// Hashes present only in `self`.
    pub only_in_self: Vec<u64>,
    /// Hashes present only in `other`.
    pub only_in_other: Vec<u64>,
}

/// An invertible Bloom lookup table: a membership sketch that can also be
/// *decoded*, listing its keys, and *subtracted*, yielding the symmetric
/// difference of two sets directly.
///
/// Each key is xor-accumulated into `k` cells with a count and a
/// verification hash. A cell whose count is `+/-1` and whose hash checks
/// out is *pure*: its key can be read off and peeled away, and peeling
/// cascades until the table is empty — if the table is big enough
/// (the paper's rule of thumb is about 1.3-1.5x the number of distinct
/// keys with `k = 4` cells per key).
///
/// ```text
/// cell = (count, key_sum, hash_sum)
/// insert("x"): cells[h_j(x)] += (1, x, H(x)) for j in 0..k
/// remove("x"): cells[h_j(x)] -= (1, x, H(x))
///
/// decode: repeat { emit a pure cell's key; remove it everywhere }
///
/// reconcile(A, B) = decode(A - B):  +1 cells are A-only, -1 are B-only
/// ```
///
/// Like every sketch in this crate, the table works on the 64-bit item
/// hash, so decoded output lists those hashes. Removing an item that was
/// never inserted corrupts the table (a residual cell fails verification
/// at decode time); only remove known-present items.
///
/// # References
///
/// - Michael T. Goodrich and Michael Mitzenmacher, "Invertible Bloom
///   Lookup Tables", Allerton 2011. <https://arxiv.org/abs/1101.2245>
/// - David Eppstein, Michael T. Goodrich, Frank Uyeda, and George
///   Varghese, "What's the Difference?: Efficient Set Reconciliation
///   Without Prior Context", SIGCOMM 2011.
///   <https://doi.org/10.1145/2018436.2018462>
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Iblt<S = DefaultBuildHasher> {
    cells: Box<[IbltCell]>,
    positions_per_key: usize,
    seed_fingerprint: u64,
    hasher: S,
}

impl Iblt<DefaultBuildHasher> {
    /// Creates an IBLT sized for an expected number of distinct keys, with
    /// hash seed zero and `k = 4` cells per key.
    ///
    /// The table holds about 1.5x `expected_distinct` cells, the paper's
    /// rule of thumb for reliable decoding.
    ///
    /// # Panics
    ///
    /// Panics if `expected_distinct` is zero.
    pub fn with_capacity(expected_distinct: u64) -> Self {
        Self::with_seed(expected_distinct, 0)
    }

    /// Creates an IBLT with an explicit hash seed.
    ///
    /// # Panics
    ///
    /// Panics if `expected_distinct` is zero.
    pub fn with_seed(expected_distinct: u64, seed: u64) -> Self {
        let hasher = DefaultBuildHasher::new(seed);
        Self::from_parts(expected_distinct, 4, hasher.seed_fingerprint(), hasher)
    }
}

impl<S> Iblt<S> {
    /// Creates an IBLT from explicit components.
    ///
    /// # Panics
    ///
    /// Panics if `expected_distinct` or `positions_per_key` is zero.
    pub fn from_parts(
        expected_distinct: u64,
        positions_per_key: usize,
        seed_fingerprint: u64,
        hasher: S,
    ) -> Self {
        assert!(
            expected_distinct > 0,
            "IBLT expected distinct count must be greater than zero"
        );
        assert!(
            positions_per_key > 0,
            "IBLT positions per key must be greater than zero"
        );
        let cells = (expected_distinct as usize * 3).div_ceil(2);
        Self {
            cells: vec![IbltCell::default(); cells].into_boxed_slice(),
            positions_per_key,
            seed_fingerprint,
            hasher,
        }
    }

    /// Returns the number of cells.
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Returns the number of cells each key maps to.
    pub const fn positions_per_key(&self) -> usize {
        self.positions_per_key
    }

    /// Returns the seed fingerprint used by merge compatibility checks.
    pub const fn seed_fingerprint(&self) -> u64 {
        self.seed_fingerprint
    }

    /// Returns the byte length of the cell storage.
    pub fn storage_bytes(&self) -> usize {
        self.cells.len() * size_of::<IbltCell>()
    }

    /// Clears all cells.
    pub fn clear(&mut self) {
        self.cells.fill(IbltCell::default());
    }

    /// The `k` cell positions of a key.
    fn positions(&self, key: u64) -> impl Iterator<Item = usize> + '_ {
        (0..self.positions_per_key).map(move |j| {
            reduce(
                mix64(key ^ (j as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)),
                self.cells.len(),
            )
        })
    }

    fn apply(&mut self, key: u64, sign: i32) {
        let verification = mix64(key);
        let positions: Vec<usize> = self.positions(key).collect();
        for position in positions {
            let cell = &mut self.cells[position];
            cell.count += sign;
            cell.key_sum ^= key;
            cell.hash_sum ^= verification;
        }
    }

    /// Peels `cells`, returning `(key, side)` for every pure cell found
    /// (`side` is the cell's count sign). Fails if peeling stalls before
    /// the table is empty.
    fn decode_cells(cells: &[IbltCell], positions_per_key: usize) -> Result<Vec<(u64, i32)>, DecodeError> {
        let mut cells = cells.to_vec();
        let mut queue: Vec<usize> = (0..cells.len())
            .filter(|&i| {
                let cell = cells[i];
                (cell.count == 1 || cell.count == -1)
                    && cell.key_sum != 0
                    && mix64(cell.key_sum) == cell.hash_sum
            })
            .collect();
        let mut out = Vec::new();

        while let Some(index) = queue.pop() {
            let cell = cells[index];
            let pure = (cell.count == 1 || cell.count == -1)
                && cell.key_sum != 0
                && mix64(cell.key_sum) == cell.hash_sum;
            if !pure {
                continue;
            }
            let side = cell.count.signum();
            let key = cell.key_sum;
            out.push((key, side));
            let verification = mix64(key);
            for position in positions_in(&cells, key, positions_per_key) {
                let target = &mut cells[position];
                target.count -= side;
                target.key_sum ^= key;
                target.hash_sum ^= verification;
                let is_pure = (target.count == 1 || target.count == -1)
                    && target.key_sum != 0
                    && mix64(target.key_sum) == target.hash_sum;
                if is_pure {
                    queue.push(position);
                }
            }
        }

        if cells
            .iter()
            .all(|cell| cell.count == 0 && cell.key_sum == 0 && cell.hash_sum == 0)
        {
            Ok(out)
        } else {
            Err(DecodeError)
        }
    }

    /// Peels the table, returning `(key, side)` for every pure cell found.
    fn decode(&self) -> Result<Vec<(u64, i32)>, DecodeError> {
        Self::decode_cells(&self.cells, self.positions_per_key)
    }
}

/// The `k` cell positions of a key in a table (free function so `decode`
/// can borrow its clone).
fn positions_in(cells: &[IbltCell], key: u64, k: usize) -> Vec<usize> {
    (0..k)
        .map(|j| reduce(mix64(key ^ (j as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)), cells.len()))
        .collect()
}

impl<S> Iblt<S>
where
    S: BuildHasher,
{
    /// Inserts `item` (its 64-bit hash, like every sketch in the crate).
    pub fn insert_item<T>(&mut self, item: &T)
    where
        T: Hash + ?Sized,
    {
        let key = hash_one(&self.hasher, item);
        self.apply(key, 1);
    }

    /// Removes one occurrence of `item`. Only remove items known to be
    /// present; removing others corrupts the table until it cannot decode.
    pub fn remove_item<T>(&mut self, item: &T)
    where
        T: Hash + ?Sized,
    {
        let key = hash_one(&self.hasher, item);
        self.apply(key, -1);
    }

    /// Returns whether `item` may be present. `false` means definitely
    /// absent (one of its cells is completely empty).
    pub fn contains_item<T>(&self, item: &T) -> bool
    where
        T: Hash + ?Sized,
    {
        let key = hash_one(&self.hasher, item);
        self.positions(key).all(|position| {
            let cell = self.cells[position];
            !(cell.count == 0 && cell.key_sum == 0 && cell.hash_sum == 0)
        })
    }

    /// Decodes the table, listing the hashes of all inserted items.
    ///
    /// Valid for insert-only tables: each key's count is positive, so the
    /// entries come back with side `+1`.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError`] when the table is too full to peel or a
    /// residual cell fails verification.
    pub fn list_entries(&self) -> Result<Vec<u64>, DecodeError> {
        Ok(self.decode()?.into_iter().map(|(key, _)| key).collect())
    }

    /// Reconciles with `other`: returns the hashes present in exactly one
    /// of the two sketches, split by side. The difference table is
    /// `self - other`.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError`] when the difference table cannot be peeled
    /// (the symmetric difference is too large for the table).
    ///
    /// # Panics
    ///
    /// Panics if the sketches differ in geometry or seed.
    pub fn reconcile(&self, other: &Self) -> Result<Reconciliation, DecodeError> {
        assert_eq!(
            self.cells.len(),
            other.cells.len(),
            "IBLT tables must have the same cell count"
        );
        assert_eq!(
            self.positions_per_key, other.positions_per_key,
            "IBLT tables must share positions per key"
        );
        assert_eq!(
            self.seed_fingerprint, other.seed_fingerprint,
            "IBLT tables must share a hash seed"
        );
        let mut difference = self.cells.clone();
        for (left, right) in difference.iter_mut().zip(other.cells.iter()) {
            left.count -= right.count;
            left.key_sum ^= right.key_sum;
            left.hash_sum ^= right.hash_sum;
        }
        let entries = Self::decode_cells(&difference, self.positions_per_key)?;
        let mut reconciliation = Reconciliation::default();
        for (key, side) in entries {
            if side > 0 {
                reconciliation.only_in_self.push(key);
            } else {
                reconciliation.only_in_other.push(key);
            }
        }
        Ok(reconciliation)
    }
}

impl<S> Sketch for Iblt<S> {
    fn clear(&mut self) {
        self.clear();
    }

    fn len_hint(&self) -> Option<u64> {
        None
    }

    fn storage_bytes(&self) -> usize {
        self.storage_bytes()
    }
}

impl<T, S> Insert<T> for Iblt<S>
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

impl<T, S> Contains<T> for Iblt<S>
where
    T: Hash + ?Sized,
    S: BuildHasher,
{
    fn contains(&self, item: &T) -> bool {
        self.contains_item(item)
    }
}

impl<T, S> Remove<T> for Iblt<S>
where
    T: Hash + ?Sized,
    S: BuildHasher,
{
    fn remove(&mut self, item: &T) -> bool {
        let was_present = self.contains_item(item);
        self.remove_item(item);
        was_present
    }
}

impl<S> Merge for Iblt<S> {
    /// Merges by cell-wise addition: the union's table. Requires equal
    /// geometry and seed.
    fn merge_from(&mut self, other: &Self) -> Result<(), MergeError> {
        if self.cells.len() != other.cells.len()
            || self.positions_per_key != other.positions_per_key
        {
            return Err(MergeError::GeometryMismatch);
        }
        if self.seed_fingerprint != other.seed_fingerprint {
            return Err(MergeError::SeedMismatch);
        }
        for (left, right) in self.cells.iter_mut().zip(other.cells.iter()) {
            left.count += right.count;
            left.key_sum ^= right.key_sum;
            left.hash_sum ^= right.hash_sum;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::Iblt;
    use crate::error::MergeError;
    use crate::traits::{Contains, Insert, Merge, Remove};

    fn hashes_of(entries: Vec<u64>, hasher: &crate::hash::DefaultBuildHasher) -> Vec<u64> {
        let mut items: Vec<u64> = (0..100).collect();
        items.retain(|item| entries.contains(&crate::hash::hash_one(hasher, item)));
        items
    }

    #[test]
    fn decode_lists_exactly_the_inserted_hashes() {
        let hasher = crate::hash::DefaultBuildHasher::new(0);
        let mut table = Iblt::with_seed(1_000, 0);
        for i in 0..100_u64 {
            table.insert_item(&i);
        }
        let entries = table.list_entries().unwrap();
        assert_eq!(entries.len(), 100);
        let found = hashes_of(entries, &hasher);
        assert_eq!(found, (0..100).collect::<Vec<u64>>());
    }

    #[test]
    fn remove_then_decode() {
        let mut table = Iblt::with_seed(1_000, 0);
        for i in 0..100_u64 {
            table.insert_item(&i);
        }
        for i in (0..100_u64).step_by(2) {
            table.remove_item(&i);
        }
        let entries = table.list_entries().unwrap();
        assert_eq!(entries.len(), 50);
        for i in (1..100_u64).step_by(2) {
            assert!(table.contains_item(&i), "missing {i} after deletions");
        }
    }

    #[test]
    fn reconcile_finds_exact_symmetric_difference() {
        let hasher = crate::hash::DefaultBuildHasher::new(0);
        let mut a = Iblt::with_seed(1_000, 0);
        let mut b = Iblt::with_seed(1_000, 0);
        for i in 0..80_u64 {
            a.insert_item(&i);
            b.insert_item(&i);
        }
        for i in 80..90_u64 {
            a.insert_item(&i);
        }
        for i in 90..100_u64 {
            b.insert_item(&i);
        }

        let reconciliation = a.reconcile(&b).unwrap();
        let only_a: Vec<u64> = (80..90).collect();
        let only_b: Vec<u64> = (90..100).collect();
        assert_eq!(hashes_of(reconciliation.only_in_self, &hasher), only_a);
        assert_eq!(hashes_of(reconciliation.only_in_other, &hasher), only_b);
    }

    #[test]
    fn overloaded_table_reports_decode_failure() {
        let mut table = Iblt::with_seed(10, 0);
        for i in 0..1_000_u64 {
            table.insert_item(&i);
        }
        assert!(table.list_entries().is_err());
    }

    #[test]
    fn merge_combines_tables_and_validates() {
        let mut left = Iblt::with_seed(1_000, 7);
        let mut right = Iblt::with_seed(1_000, 7);
        for i in 0..50_u64 {
            left.insert_item(&i);
        }
        for i in 50..100_u64 {
            right.insert_item(&i);
        }
        left.merge_from(&right).unwrap();
        assert_eq!(left.list_entries().unwrap().len(), 100);

        let other_seed = Iblt::with_seed(1_000, 8);
        assert_eq!(left.merge_from(&other_seed), Err(MergeError::SeedMismatch));
    }

    #[test]
    fn capability_traits_work() {
        let mut table = Iblt::with_seed(100, 0);
        Insert::<u64>::insert(&mut table, &7).unwrap();
        assert!(Contains::<u64>::contains(&table, &7));
        assert!(Remove::<u64>::remove(&mut table, &7));
    }
}
