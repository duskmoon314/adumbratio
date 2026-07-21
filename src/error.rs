//! Error types shared by sketch implementations.

use std::fmt;

/// Error returned when two sketches cannot be merged safely.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MergeError {
    /// The sketches use different geometry, such as bit count or row width.
    GeometryMismatch,
    /// The sketches were built with different hash seeds.
    SeedMismatch,
}

impl fmt::Display for MergeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GeometryMismatch => f.write_str("sketch geometry does not match"),
            Self::SeedMismatch => f.write_str("sketch seed does not match"),
        }
    }
}

impl std::error::Error for MergeError {}

/// Error returned when a bucket cannot accept another fingerprint.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BucketFull;

impl fmt::Display for BucketFull {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("bucket is full")
    }
}

impl std::error::Error for BucketFull {}

/// Error returned when a bounded sketch cannot accept another item.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SketchFull {
    orphaned_fingerprint: u64,
}

impl SketchFull {
    /// Creates an insertion-capacity error with the displaced fingerprint.
    pub const fn new(orphaned_fingerprint: u64) -> Self {
        Self {
            orphaned_fingerprint,
        }
    }

    /// Returns the fingerprint that could not be placed.
    pub const fn orphaned_fingerprint(self) -> u64 {
        self.orphaned_fingerprint
    }
}

impl fmt::Display for SketchFull {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("sketch is full")
    }
}

impl std::error::Error for SketchFull {}
