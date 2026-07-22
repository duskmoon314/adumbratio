use alloc::boxed::Box;
use alloc::vec;

/// A packed array of fixed-width unsigned cells.
///
/// Cells are packed tightly into `u64` words and may straddle word boundaries.
/// The cell width is a const generic so arrays with different widths are
/// distinct types.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PackedArray<const BITS: u32> {
    words: Box<[u64]>,
    len: usize,
}

impl<const BITS: u32> PackedArray<BITS> {
    /// The maximum value representable by a cell of this width.
    pub const MAX: u64 = if BITS == u64::BITS {
        u64::MAX
    } else if BITS == 0 || BITS > u64::BITS {
        0
    } else {
        (1_u64 << BITS) - 1
    };

    /// Creates an array containing `len` zero-valued cells.
    ///
    /// # Panics
    ///
    /// Panics if `BITS` is not in `1..=64` or if the required bit length
    /// overflows `usize`.
    pub fn new(len: usize) -> Self {
        Self::check_width();
        let words = if BITS == u64::BITS {
            len
        } else {
            let bits = len
                .checked_mul(BITS as usize)
                .expect("packed array bit length overflowed usize");
            bits.div_ceil(u64::BITS as usize)
        };

        Self {
            words: vec![0; words].into_boxed_slice(),
            len,
        }
    }

    /// Returns the number of addressable cells.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the array contains no cells.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the value at cell `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.len()`.
    pub fn get(&self, i: usize) -> u64 {
        self.check_index(i);
        if BITS == u64::BITS {
            return self.words[i];
        }

        let start = i * BITS as usize;
        let word = start / u64::BITS as usize;
        let offset = start % u64::BITS as usize;
        let bits = BITS as usize;

        if offset + bits <= u64::BITS as usize {
            (self.words[word] >> offset) & Self::MAX
        } else {
            let low_bits = u64::BITS as usize - offset;
            let high_bits = bits - low_bits;
            let low = self.words[word] >> offset;
            let high = self.words[word + 1] & low_mask(high_bits);
            low | (high << low_bits)
        }
    }

    /// Sets cell `i` to `v`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.len()` or if `v > Self::MAX`.
    pub fn set(&mut self, i: usize, v: u64) {
        self.check_index(i);
        assert!(
            v <= Self::MAX,
            "packed value {v} exceeds {}-bit maximum {}",
            BITS,
            Self::MAX
        );

        if BITS == u64::BITS {
            self.words[i] = v;
            return;
        }

        let start = i * BITS as usize;
        let word = start / u64::BITS as usize;
        let offset = start % u64::BITS as usize;
        let bits = BITS as usize;

        if offset + bits <= u64::BITS as usize {
            let mask = Self::MAX << offset;
            self.words[word] = (self.words[word] & !mask) | (v << offset);
        } else {
            let low_bits = u64::BITS as usize - offset;
            let high_bits = bits - low_bits;
            let low_word_mask = u64::MAX << offset;
            let high_mask = low_mask(high_bits);

            self.words[word] = (self.words[word] & !low_word_mask) | (v << offset);
            self.words[word + 1] =
                (self.words[word + 1] & !high_mask) | ((v >> low_bits) & high_mask);
        }
    }

    /// Applies `f` to cell `i`, stores the returned value, and returns it.
    ///
    /// # Panics
    ///
    /// Panics under the same conditions as [`Self::get`] and [`Self::set`].
    pub fn update(&mut self, i: usize, f: impl FnOnce(u64) -> u64) -> u64 {
        let next = f(self.get(i));
        self.set(i, next);
        next
    }

    /// Combines this array with `other` cell-by-cell.
    ///
    /// # Panics
    ///
    /// Panics if the arrays have different lengths or if `f` returns a value
    /// outside the cell range.
    pub fn merge_with(&mut self, other: &Self, f: impl Fn(u64, u64) -> u64) {
        assert_eq!(
            self.len, other.len,
            "packed arrays must have the same length"
        );
        for i in 0..self.len {
            self.set(i, f(self.get(i), other.get(i)));
        }
    }

    /// Clears every cell to zero.
    pub fn clear(&mut self) {
        self.words.fill(0);
    }

    /// Returns the byte length of the backing word storage.
    pub fn storage_bytes(&self) -> usize {
        self.words.len() * size_of::<u64>()
    }

    fn check_width() {
        assert!(
            (1..=u64::BITS).contains(&BITS),
            "packed array width must be in 1..=64 bits"
        );
    }

    fn check_index(&self, i: usize) {
        Self::check_width();
        assert!(
            i < self.len,
            "packed index {i} out of bounds for packed array of length {}",
            self.len
        );
    }
}

fn low_mask(bits: usize) -> u64 {
    if bits == u64::BITS as usize {
        u64::MAX
    } else {
        (1_u64 << bits) - 1
    }
}

#[cfg(test)]
mod tests {
    use super::PackedArray;

    #[test]
    fn stores_one_bit_cells() {
        let mut cells = PackedArray::<1>::new(129);
        cells.set(0, 1);
        cells.set(128, 1);

        assert_eq!(cells.get(0), 1);
        assert_eq!(cells.get(1), 0);
        assert_eq!(cells.get(128), 1);
    }

    #[test]
    fn stores_cross_word_cells() {
        let mut cells = PackedArray::<6>::new(24);
        cells.set(10, 0b10_1011);
        cells.set(11, 0b01_0110);

        assert_eq!(cells.get(9), 0);
        assert_eq!(cells.get(10), 0b10_1011);
        assert_eq!(cells.get(11), 0b01_0110);
    }

    #[test]
    fn stores_common_counter_widths() {
        let mut four = PackedArray::<4>::new(20);
        four.set(3, 15);
        four.set(4, 7);
        assert_eq!(four.get(3), 15);
        assert_eq!(four.get(4), 7);

        let mut thirty_two = PackedArray::<32>::new(4);
        thirty_two.set(2, u32::MAX as u64);
        assert_eq!(thirty_two.get(2), u32::MAX as u64);

        let mut sixty_four = PackedArray::<64>::new(2);
        sixty_four.set(1, u64::MAX - 9);
        assert_eq!(sixty_four.get(1), u64::MAX - 9);
    }

    #[test]
    fn update_and_merge_cells() {
        let mut left = PackedArray::<4>::new(8);
        let mut right = PackedArray::<4>::new(8);
        left.set(3, 4);
        right.set(3, 5);

        assert_eq!(left.update(3, |v| v + 1), 5);
        left.merge_with(&right, |a, b| a + b);
        assert_eq!(left.get(3), 10);
    }
}
