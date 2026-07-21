//! Composed sketches.

mod bloom;
mod counting_bloom;

pub use bloom::{BloomFilter, BloomGeometry};
pub use counting_bloom::CountingBloomFilter;
