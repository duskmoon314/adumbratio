//! Eviction support for Cuckoo filters.

/// Minimal random-number interface used by storage and eviction code.
pub trait RngLite {
    /// Returns the next pseudorandom `u64`.
    fn next_u64(&mut self) -> u64;

    /// Returns a pseudorandom index in `0..upper`.
    ///
    /// # Panics
    ///
    /// Panics if `upper` is zero.
    fn next_index(&mut self, upper: usize) -> usize {
        assert!(upper > 0, "random index upper bound must be non-zero");
        crate::hash::reduce(self.next_u64(), upper)
    }
}

/// Small deterministic xorshift generator used by default.
///
/// This generator is for repeatable sketch eviction only. It is not
/// cryptographically secure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    /// Creates a generator from a seed.
    pub const fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            seed
        };
        Self { state }
    }
}

impl Default for XorShift64 {
    fn default() -> Self {
        Self::new(0)
    }
}

impl RngLite for XorShift64 {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

/// Bound for the Cuckoo filter displacement loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KickLoop {
    /// Maximum number of evictions attempted by one insertion.
    pub max_kicks: usize,
}

impl KickLoop {
    /// Creates a kick-loop policy with an explicit bound.
    pub const fn new(max_kicks: usize) -> Self {
        Self { max_kicks }
    }
}

impl Default for KickLoop {
    fn default() -> Self {
        Self { max_kicks: 500 }
    }
}
