//! Composable building blocks for probabilistic data structures.
//!
//! `adumbratio` decomposes sketches into storage, hashing, policy, and
//! composed-sketch layers. The ready-made sketches — Bloom, counting Bloom,
//! blocked Bloom, Count-Min, Count Sketch, Cuckoo filter, HyperLogLog,
//! MinHash, and a top-k heavy-hitters companion — are built entirely from
//! the public block APIs, and custom compositions can be assembled from the
//! same blocks.
//!
//! The crate is `no_std` (+`alloc`) compatible: disable default features
//! for the core operations, and enable the optional `libm` feature when
//! geometry solvers and error estimators are needed without `std`.
//!
//! ```
//! use adumbratio::sketch::BloomFilter;
//!
//! let mut filter = BloomFilter::with_capacity(1_000, 0.01);
//! filter.insert_item("alice");
//!
//! assert!(filter.contains_item("alice"));
//! assert!(!filter.contains_item("bob"));
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]
#![forbid(unsafe_code)]

extern crate alloc;

#[cfg(any(feature = "std", feature = "libm"))]
mod float;

pub mod block;
pub mod error;
pub mod hash;
pub mod policy;
pub mod sketch;
pub mod traits;

pub use error::{DecodeError, MergeError};
