use alloc::boxed::Box;
use alloc::vec;

/// A packed array of single-bit cells.
///
/// `BitArray` has no hashing or probabilistic semantics. It is the storage
/// block used by Bloom-style sketches and by custom compositions that need a
/// dense set of boolean cells.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BitArray {
    words: Box<[u64]>,
    len: usize,
}

impl BitArray {
    /// Creates an array containing `len` cleared bits.
    pub fn new(len: usize) -> Self {
        let words = len.div_ceil(u64::BITS as usize);
        Self {
            words: vec![0; words].into_boxed_slice(),
            len,
        }
    }

    /// Returns the number of addressable bits.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the array contains no bits.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the bit at index `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.len()`.
    pub fn get(&self, i: usize) -> bool {
        self.check_index(i);
        let word = i / u64::BITS as usize;
        let bit = i % u64::BITS as usize;
        (self.words[word] & (1_u64 << bit)) != 0
    }

    /// Sets the bit at index `i`, returning its previous value.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.len()`.
    pub fn set(&mut self, i: usize) -> bool {
        self.check_index(i);
        let word = i / u64::BITS as usize;
        let bit = i % u64::BITS as usize;
        let mask = 1_u64 << bit;
        let previous = (self.words[word] & mask) != 0;
        self.words[word] |= mask;
        previous
    }

    /// Clears the bit at index `i`, returning its previous value.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.len()`.
    pub fn unset(&mut self, i: usize) -> bool {
        self.check_index(i);
        let word = i / u64::BITS as usize;
        let bit = i % u64::BITS as usize;
        let mask = 1_u64 << bit;
        let previous = (self.words[word] & mask) != 0;
        self.words[word] &= !mask;
        previous
    }

    /// Counts the set bits in the array.
    pub fn count_ones(&self) -> usize {
        self.words
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    /// Replaces this array with the bitwise union of `self` and `other`.
    ///
    /// # Panics
    ///
    /// Panics if the arrays have different lengths.
    pub fn union_with(&mut self, other: &Self) {
        self.check_same_len(other);
        for (left, right) in self.words.iter_mut().zip(other.words.iter()) {
            *left |= *right;
        }
    }

    /// Replaces this array with the bitwise intersection of `self` and `other`.
    ///
    /// # Panics
    ///
    /// Panics if the arrays have different lengths.
    pub fn intersect_with(&mut self, other: &Self) {
        self.check_same_len(other);
        for (left, right) in self.words.iter_mut().zip(other.words.iter()) {
            *left &= *right;
        }
    }

    /// Clears every bit.
    pub fn clear(&mut self) {
        self.words.fill(0);
    }

    /// Returns the byte length of the backing word storage.
    pub fn storage_bytes(&self) -> usize {
        self.words.len() * size_of::<u64>()
    }

    fn check_index(&self, i: usize) {
        assert!(
            i < self.len,
            "bit index {i} out of bounds for bit array of length {}",
            self.len
        );
    }

    fn check_same_len(&self, other: &Self) {
        assert_eq!(self.len, other.len, "bit arrays must have the same length");
    }
}

#[cfg(test)]
mod tests {
    use super::BitArray;

    #[test]
    fn set_get_count_and_clear_bits() {
        let mut bits = BitArray::new(130);

        assert!(!bits.get(0));
        assert!(!bits.set(0));
        assert!(bits.set(0));
        assert!(!bits.set(64));
        assert!(!bits.set(129));

        assert!(bits.get(0));
        assert!(bits.get(64));
        assert!(bits.get(129));
        assert_eq!(bits.count_ones(), 3);

        bits.clear();
        assert_eq!(bits.count_ones(), 0);
        assert!(!bits.get(0));
    }

    #[test]
    fn union_and_intersection_match_naive_sets() {
        let mut left = BitArray::new(96);
        let mut right = BitArray::new(96);

        for i in [1, 2, 63, 95] {
            left.set(i);
        }
        for i in [2, 63, 64] {
            right.set(i);
        }

        let mut union = left.clone();
        union.union_with(&right);
        for i in [1, 2, 63, 64, 95] {
            assert!(union.get(i));
        }
        assert_eq!(union.count_ones(), 5);

        left.intersect_with(&right);
        assert!(left.get(2));
        assert!(left.get(63));
        assert_eq!(left.count_ones(), 2);
    }
}
