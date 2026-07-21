//! Update policies.

mod counter;
mod eviction;
mod update;

pub use counter::{Checked, CounterPolicy, Saturating};
pub use eviction::{KickLoop, RngLite, XorShift64};
pub use update::{ConservativeUpdate, PlainUpdate};
