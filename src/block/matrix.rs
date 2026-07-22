use alloc::vec::Vec;

/// A fixed-width collection of equally shaped storage rows.
///
/// `Matrix` is intentionally thin: it owns row objects and records the row
/// width that indexing schemes use. It does not impose cell semantics.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Matrix<A> {
    rows: Vec<A>,
    width: usize,
}

impl<A> Matrix<A> {
    /// Creates a matrix from pre-built rows.
    ///
    /// # Panics
    ///
    /// Panics if `rows` is empty.
    pub fn from_rows(rows: Vec<A>, width: usize) -> Self {
        assert!(!rows.is_empty(), "matrix must contain at least one row");
        Self { rows, width }
    }

    /// Returns the number of rows.
    pub fn rows(&self) -> usize {
        self.rows.len()
    }

    /// Returns the declared width of each row.
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Returns an immutable row by index.
    ///
    /// # Panics
    ///
    /// Panics if `row` is out of bounds.
    pub fn row(&self, row: usize) -> &A {
        &self.rows[row]
    }

    /// Returns a mutable row by index.
    ///
    /// # Panics
    ///
    /// Panics if `row` is out of bounds.
    pub fn row_mut(&mut self, row: usize) -> &mut A {
        &mut self.rows[row]
    }

    /// Iterates over rows.
    pub fn iter(&self) -> impl Iterator<Item = &A> {
        self.rows.iter()
    }

    /// Iterates mutably over rows.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut A> {
        self.rows.iter_mut()
    }
}
