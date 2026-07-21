//! Composable building blocks for probabilistic data structures.
//!
//! `adumbratio` decomposes sketches into storage, hashing, policy, and
//! composed-sketch layers. All sketches speak a shared vocabulary of
//! capability traits.

#![deny(missing_docs)]

pub mod block;
pub mod error;
pub mod traits;

pub use error::MergeError;
