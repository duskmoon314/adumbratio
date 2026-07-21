//! Hashing blocks.

mod fingerprint;
mod index;
mod item;

pub use fingerprint::PartialKeyCuckoo;
pub use index::{
    Blocked, DoubleHashing, EnhancedDoubleHashing, IndexScheme, Partitioned, reduce, row_index,
    sign,
};
pub use item::{DefaultBuildHasher, ItemHasher, StableHasher, hash_one};

pub(crate) fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}
