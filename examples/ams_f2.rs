//! Reproduces the F2 (self-join size) estimation of the AMS tug-of-war.
//!
//! # What the paper establishes
//!
//! - Noga Alon, Yossi Matias, and Mario Szegedy, "The Space Complexity of
//!   Approximating the Frequency Moments", STOC 1996 (journal: JCSS 1999).
//!   <https://doi.org/10.1145/237814.237823>
//!   The founding paper of sketching. For the second frequency moment
//!   `F2 = sum(f_i^2)`, the tug-of-war estimator keeps signed sums
//!   `z = sum(sigma_i f_i)` with random `sigma_i in {-1, +1}`; `z^2` is
//!   unbiased for F2, and a median-of-means over groups of counters brings
//!   the relative error below `epsilon` with probability `1 - delta` using
//!   `O((1/eps^2) * log(1/delta))` counters. The paper's other equally
//!   important point: the sketch is a *linear map* of the stream, so it
//!   merges exactly — the property that made sketches composable systems
//!   components rather than one-off algorithms.
//!
//! # What this example does
//!
//! 1. Streams a Zipf(1.1) distribution (where F2 is dominated by a few
//!    heavy hitters, the interesting regime) and prints the F2 estimate
//!    against the exact value at several widths — the error should shrink
//!    like `1/width^{1/2}`.
//! 2. Demonstrates exact linear merging: counters of merged halves equal
//!    the counters of one sketch over the whole stream, so merged and
//!    direct F2 estimates are identical.
//!
//! The stream is driven by a seeded xorshift, so the output is
//! reproducible. Run with: `cargo run --release --example ams_f2`

use adumbratio::policy::{RngLite, XorShift64};
use adumbratio::sketch::AmsSketch;
use adumbratio::traits::Merge;

const EVENTS: usize = 200_000;

fn main() {
    println!("adumbratio — AMS tug-of-war F2 estimation (Alon–Matias–Szegedy 1996)");
    println!("N = {EVENTS} events, Zipf(1.1) over 10000 distinct items\n");

    let counts = zipf_counts(10_000, EVENTS, 1.1, 11);
    let truth: f64 = counts.iter().map(|&c| (c as f64) * (c as f64)).sum();
    println!("exact F2 = {truth:.0}\n");

    // -- Part 1: estimate vs. width ------------------------------------------------
    println!("Part 1: F2 estimate as the width grows (4 groups)");
    println!("{:<12} {:>16} {:>14}", "width", "estimate", "rel. error");
    for width in [64_usize, 256, 1_024, 4_096] {
        let mut sketch = AmsSketch::with_seed(4, width, 5);
        for (item, &count) in counts.iter().enumerate() {
            for _ in 0..count {
                sketch.insert_item(&(item as u64));
            }
        }
        let estimate = sketch.f2();
        let rel = (estimate - truth).abs() / truth;
        println!("{:<12} {:>16.0} {:>13.2}%", width, estimate, rel * 100.0);
    }
    println!("paper's scaling: error ~ sqrt(2/width)\n");

    // -- Part 2: exact linear merge -------------------------------------------------
    println!("Part 2: linear merge is exact");
    let mut left = AmsSketch::with_seed(4, 256, 5);
    let mut right = AmsSketch::with_seed(4, 256, 5);
    let mut single = AmsSketch::with_seed(4, 256, 5);
    for (item, &count) in counts.iter().enumerate() {
        for _ in 0..count {
            if item % 2 == 0 {
                left.insert_item(&(item as u64));
            } else {
                right.insert_item(&(item as u64));
            }
            single.insert_item(&(item as u64));
        }
    }
    left.merge_from(&right).unwrap();
    println!(
        "merged halves vs. whole stream: identical counters = {}, identical F2 = {}",
        left.counters() == single.counters(),
        left.f2() == single.f2()
    );
}

/// Simulates a Zipf(s) stream and returns true per-item frequencies, the
/// same construction used in the Count-Min example.
fn zipf_counts(universe: u64, events: usize, s: f64, seed: u64) -> Vec<u64> {
    let mut cumulative = Vec::with_capacity(universe as usize);
    let mut acc = 0.0_f64;
    for i in 0..universe {
        acc += 1.0 / ((i + 1) as f64).powf(s);
        cumulative.push(acc);
    }
    let total = acc;

    let mut rng = XorShift64::new(seed);
    let mut counts = vec![0_u64; universe as usize];
    for _ in 0..events {
        let target = (rng.next_u64() as f64 / u64::MAX as f64) * total;
        let index = cumulative.partition_point(|&c| c < target);
        counts[index.min(universe as usize - 1)] += 1;
    }
    counts
}
