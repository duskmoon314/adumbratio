//! Capability traits shared by sketches.

/// Common lifecycle and storage introspection for sketches.
///
/// Lifecycle notes for stream-processing consumers:
///
/// - [`Sketch::clear`] empties the sketch but keeps its allocation and
///   geometry (it is the "reset" of windowed pipelines — reuse the same
///   object for the next window).
/// - Merging is safe only between sketches with identical geometry *and*
///   seed: implementations compare a stored seed fingerprint and report
///   [`crate::error::MergeError::SeedMismatch`] (or
///   [`crate::error::MergeError::GeometryMismatch`]) instead of silently
///   corrupting answers.
/// - Geometry accessors (`geometry()`, `storage_bytes()`, and friends)
///   report the realized parameters, so a cleared sketch is cheap to
///   re-validate or rebuild identically.
pub trait Sketch {
    /// Clears all sketch state while retaining its allocation and geometry.
    fn clear(&mut self);

    /// Returns an approximate stored-item count when the sketch can provide one.
    fn len_hint(&self) -> Option<u64>;

    /// Returns the number of bytes used by the sketch storage layer.
    fn storage_bytes(&self) -> usize;
}

/// Capability for sketches that can ingest items.
pub trait Insert<T: ?Sized> {
    /// Error type returned by insertion.
    type Err;

    /// Inserts one occurrence of `item`.
    fn insert(&mut self, item: &T) -> Result<(), Self::Err>;
}

/// Capability for sketches that can answer membership queries.
pub trait Contains<T: ?Sized> {
    /// Returns whether `item` may be present.
    ///
    /// A `false` result means definitely absent for membership sketches.
    fn contains(&self, item: &T) -> bool;
}

/// Capability for sketches that can remove items.
pub trait Remove<T: ?Sized> {
    /// Removes one occurrence of `item` when the sketch supports deletion.
    fn remove(&mut self, item: &T) -> bool;
}

/// Capability for sketches that can estimate point frequencies.
pub trait EstimateCount<T: ?Sized> {
    /// Returns the estimated frequency for `item`.
    fn estimate(&self, item: &T) -> u64;
}

/// Capability for sketches that can estimate total cardinality.
pub trait EstimateCardinality {
    /// Returns an approximate number of distinct inserted items.
    fn cardinality(&self) -> f64;
}

/// Capability for sketches that can absorb another sketch of the same type.
pub trait Merge: Sized {
    /// Merges `other` into `self`.
    ///
    /// Implementations return an error if the sketches do not share compatible
    /// geometry and hash seeds.
    fn merge_from(&mut self, other: &Self) -> Result<(), crate::error::MergeError>;
}

/// Capability for frequency sketches that can back a top-k candidate set:
/// point estimates and weighted inserts on top of the usual lifecycle and
/// merge vocabulary.
///
/// Implemented by [`CountMinSketch`](crate::sketch::CountMinSketch) and
/// [`CountSketch`](crate::sketch::CountSketch), which is what makes
/// [`TopK`](crate::sketch::TopK) work over either backend.
pub trait Estimator<T: ?Sized>: Sketch + Merge {
    /// Estimates the frequency of `item`.
    fn estimate(&self, item: &T) -> u64;

    /// Inserts `item` with weight `count`.
    fn insert_count(&mut self, item: &T, count: u64);

    /// Returns the total number of inserted events (gross weighted volume).
    fn total(&self) -> u64;
}
