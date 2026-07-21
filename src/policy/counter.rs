//! Counter arithmetic policies.

/// Arithmetic behavior for packed unsigned counters.
pub trait CounterPolicy {
    /// Returns the result of adding `by` to `cell`.
    fn increment(&self, cell: u64, max: u64, by: u64) -> u64;

    /// Returns the result of subtracting `by` from `cell`.
    fn decrement(&self, cell: u64, max: u64, by: u64) -> u64;

    /// Combines two counter cells during merge.
    fn merge(&self, a: u64, b: u64, max: u64) -> u64;
}

/// Saturating counter arithmetic.
///
/// Increments and merges pin at `max`. Decrementing a counter already at
/// `max` leaves it at `max`, which preserves the counting-Bloom invariant that
/// saturated counters are sticky and cannot create false negatives.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Saturating;

impl CounterPolicy for Saturating {
    fn increment(&self, cell: u64, max: u64, by: u64) -> u64 {
        cell.saturating_add(by).min(max)
    }

    fn decrement(&self, cell: u64, max: u64, by: u64) -> u64 {
        if cell == max {
            max
        } else {
            cell.saturating_sub(by)
        }
    }

    fn merge(&self, a: u64, b: u64, max: u64) -> u64 {
        a.saturating_add(b).min(max)
    }
}

/// Checked counter arithmetic.
///
/// Operations panic on overflow or underflow instead of saturating.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Checked;

impl CounterPolicy for Checked {
    fn increment(&self, cell: u64, max: u64, by: u64) -> u64 {
        assert!(cell <= max, "counter cell exceeds maximum");
        let next = cell.checked_add(by).expect("counter increment overflowed");
        assert!(next <= max, "counter increment exceeds maximum");
        next
    }

    fn decrement(&self, cell: u64, max: u64, by: u64) -> u64 {
        assert!(cell <= max, "counter cell exceeds maximum");
        assert!(by <= cell, "counter decrement underflowed");
        cell - by
    }

    fn merge(&self, a: u64, b: u64, max: u64) -> u64 {
        assert!(a <= max && b <= max, "counter cell exceeds maximum");
        let next = a.checked_add(b).expect("counter merge overflowed");
        assert!(next <= max, "counter merge exceeds maximum");
        next
    }
}

#[cfg(test)]
mod tests {
    use super::{CounterPolicy, Saturating};

    #[test]
    fn saturating_counter_is_sticky_at_maximum() {
        let policy = Saturating;
        assert_eq!(policy.increment(14, 15, 10), 15);
        assert_eq!(policy.decrement(15, 15, 1), 15);
        assert_eq!(policy.decrement(8, 15, 3), 5);
    }
}
