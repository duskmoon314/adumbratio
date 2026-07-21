//! Composable building blocks for probabilistic data structures.
//!
//! `adumbratio` decomposes sketches into storage, hashing, policy, and
//! composed-sketch layers. The first implemented sketch is a Bloom filter built
//! entirely from the public block APIs.
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

#![deny(missing_docs)]

pub mod block;
pub mod error;
pub mod hash;
pub mod policy;
pub mod sketch;
pub mod traits;

pub use error::MergeError;
