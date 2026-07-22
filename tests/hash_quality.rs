//! Statistical quality tests for the vendored stable hash.
//!
//! Every guarantee in the crate (FPP bounds, error bounds, mergeability)
//! rests on `StableHasher` producing uniform, avalanche-complete 64-bit
//! hashes. These tests check the two standard properties:
//!
//! - **Avalanche:** flipping any one input bit flips each output bit with
//!   probability 1/2 (bit independence, checked over the full 64x64
//!   input/output bit matrix).
//! - **Uniformity on correlated inputs:** sequential integers — the worst
//!   case for sketch workloads, since users insert IDs — produce uniform
//!   output bits and a flat bucket distribution under `reduce()`.
//!
//! Everything is seeded, so the suite is deterministic. Tolerances are
//! 4-sigma bounds of the binomial/chi-square null distributions, so a
//! correct mixer passes with overwhelming margin while a broken one (say,
//! a missing finalizer round) fails decisively.

use adumbratio::hash::{DefaultBuildHasher, hash_one, reduce};
use adumbratio::policy::{RngLite, XorShift64};

const SAMPLES: u64 = 20_000;

fn hash(item: u64) -> u64 {
    hash_one(&DefaultBuildHasher::new(42), &item)
}

/// Returns the 64x64 flip-probability matrix: entry [b][j] is the observed
/// probability that output bit j flips when input bit b is flipped. Each
/// input bit gets an independent base sample, decorrelating the cells.
// Bit positions are semantically meaningful here, so range loops over
// `enumerate` are the clearer form.
#[allow(clippy::needless_range_loop)]
fn avalanche_matrix() -> [[f64; 64]; 64] {
    let mut matrix = [[0.0_f64; 64]; 64];
    for b in 0..64 {
        let mut rng = XorShift64::new(7 + b as u64);
        for _ in 0..SAMPLES {
            let base = rng.next_u64();
            let delta = hash(base) ^ hash(base ^ (1 << b));
            for j in 0..64 {
                if delta >> j & 1 == 1 {
                    matrix[b][j] += 1.0;
                }
            }
        }
        for j in 0..64 {
            matrix[b][j] /= SAMPLES as f64;
        }
    }
    matrix
}

#[test]
fn hash_has_full_avalanche() {
    let matrix = avalanche_matrix();
    // Binomial(20000, 0.5) has sigma ~0.0035 in probability. The expected
    // maximum deviation over the 4096 cells is ~3.9 sigma; the 0.018
    // tolerance is a 5-sigma bound, passing a correct mixer with
    // overwhelming margin.
    for (b, row) in matrix.iter().enumerate() {
        for (j, &p) in row.iter().enumerate() {
            assert!(
                (0.482..=0.518).contains(&p),
                "input bit {b} -> output bit {j}: flip probability {p} outside 0.5 +/- 0.018"
            );
        }
    }
}

#[test]
// Bit positions are semantically meaningful in this loop.
#[allow(clippy::needless_range_loop)]
fn hash_uniform_on_sequential_inputs() {
    // Sequential IDs are the canonical sketch input; per-bit means must stay
    // near 0.5. sigma for 100k samples is ~0.0016; 0.01 is a 6-sigma bound.
    let n = 100_000_u64;
    let mut ones = [0_u64; 64];
    for i in 0..n {
        let h = hash(i);
        for j in 0..64 {
            ones[j] += h >> j & 1;
        }
    }
    for (j, &count) in ones.iter().enumerate() {
        let mean = count as f64 / n as f64;
        assert!(
            (0.49..=0.51).contains(&mean),
            "output bit {j}: mean {mean} outside 0.5 +/- 0.01 on sequential inputs"
        );
    }
}

#[test]
fn reduction_distributes_uniformly_over_buckets() {
    // Chi-square goodness-of-fit for reduce() into 1024 buckets over 100k
    // sequential items: mean 97.7 per bucket, chi2 ~ 1023 +/- 45 at 1 sigma.
    let buckets = 1024_usize;
    let n = 100_000_u64;
    let mut counts = vec![0_u64; buckets];
    for i in 0..n {
        counts[reduce(hash(i), buckets)] += 1;
    }
    let expected = n as f64 / buckets as f64;
    let chi2: f64 = counts
        .iter()
        .map(|&c| (c as f64 - expected).powi(2) / expected)
        .sum();
    assert!(
        chi2 < 1023.0 + 5.0 * 45.2,
        "chi-square {chi2} exceeds the 5-sigma uniform bound"
    );
}
