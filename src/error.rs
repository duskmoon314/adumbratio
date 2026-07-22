//! Error types shared by sketch implementations.

use core::fmt;

/// Error returned when two sketches cannot be merged safely.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MergeError {
    /// The sketches use different geometry, such as bit count or row width.
    GeometryMismatch,
    /// The sketches were built with different hash seeds.
    SeedMismatch,
    /// Merging succeeded in principle, but the destination lacks the
    /// capacity to hold the union (e.g. a quotient filter filling up).
    InsufficientCapacity,
}

impl fmt::Display for MergeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GeometryMismatch => f.write_str("sketch geometry does not match"),
            Self::SeedMismatch => f.write_str("sketch seed does not match"),
            Self::InsufficientCapacity => {
                f.write_str("insufficient capacity to hold the merged union")
            }
        }
    }
}

impl core::error::Error for MergeError {}

/// Error returned when a bucket cannot accept another fingerprint.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BucketFull;

impl fmt::Display for BucketFull {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("bucket is full")
    }
}

impl core::error::Error for BucketFull {}

/// Error returned when a bounded sketch cannot accept another item.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

impl core::error::Error for SketchFull {}
