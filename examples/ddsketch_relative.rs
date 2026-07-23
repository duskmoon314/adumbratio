//! Reproduces DDSketch's relative-error guarantee across the quantile range.
//!
//! # What the paper evaluates
//!
//! - Charles Masson, Jee E. Rim, and Homin K. Lee, "DDSketch: A Fast and
//!   Fully-Mergeable Quantile Sketch with Relative-Error Guarantees",
//!   PVLDB 2019. <https://doi.org/10.14778/3342263.3342635>
//!   The paper's promise: quantile estimates within a factor
//!   `1 +/- alpha` of the true value *everywhere on the range* — the p99.9
//!   of a heavy-tailed latency distribution is as accurate, relatively, as
//!   the median. Absolute-error sketches (KLL, GK) cannot offer this: an
//!   `eps * N` rank error translates to enormous relative value error in
//!   the tail. Their evaluation measures the relative value error across
//!   quantiles on synthetic and real distributions, and mergeability in
//!   distributed settings. The second table below is exactly that first
//!   measurement.
//!
//! # What this example does
//!
//! 1. Streams log-normal values (heavy-tailed, the latency shape) and
//!    prints estimate/truth ratios from p50 to p99.9 — every row must sit
//!    inside `1 +/- alpha`.
//! 2. Compares with the KLL sketch at matched error parameters: fine at
//!    the median, useless relatively in the tail — the contrast the paper
//!    is about.
//! 3. Demonstrates exact merging of two half-streams.
//!
//! Streams are driven by a seeded xorshift, so the output is
//! reproducible. Run with: `cargo run --release --example ddsketch_relative`

use adumbratio::policy::{RngLite, XorShift64};
use adumbratio::sketch::{DdSketch, KllSketch};
use adumbratio::traits::Merge;

/// Total-order wrapper for f64 (KLL requires Ord; values are finite here).
#[derive(Clone, Copy, PartialEq)]
struct Ord64(f64);

impl Eq for Ord64 {}

impl PartialOrd for Ord64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ord64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Log-normal sample via the Box-Muller transform on xorshift output.
fn log_normal(rng: &mut XorShift64) -> f64 {
    let u1 = (rng.next_u64() as f64 + 1.0) / u64::MAX as f64;
    let u2 = rng.next_u64() as f64 / u64::MAX as f64;
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    (1.0 + z).exp()
}

const N: usize = 200_000;

fn main() {
    println!("adumbratio — DDSketch relative error (Masson–Rim–Lee 2019)");
    println!("N = {N} log-normal values, alpha = 2%\n");

    let mut rng = XorShift64::new(13);
    let mut dd = DdSketch::new(0.02);
    let mut kll = KllSketch::with_seed(200, 1);
    let mut values = Vec::with_capacity(N);
    for _ in 0..N {
        let v = log_normal(&mut rng);
        dd.insert_item(&v);
        kll.insert_item(&Ord64(v));
        values.push(v);
    }
    values.sort_by(f64::total_cmp);

    // -- Part 1: ratio across quantiles -------------------------------------------
    println!("Part 1: DDSketch estimate/truth ratio (guarantee: 1 +/- 0.02)");
    println!(
        "{:<10} {:>14} {:>14} {:>12}",
        "q", "truth", "estimate", "ratio"
    );
    for q in [0.5, 0.9, 0.99, 0.999] {
        let truth = values[(q * (N - 1) as f64) as usize];
        let estimate = dd.quantile(q).unwrap();
        println!(
            "{:<10.3} {:>14.3} {:>14.3} {:>12.4}",
            q,
            truth,
            estimate,
            estimate / truth
        );
    }

    // -- Part 2: vs. absolute-error sketches in the tail -----------------------------
    println!("\nPart 2: same data, KLL (absolute rank error) for contrast");
    println!("{:<10} {:>14} {:>14}", "q", "KLL estimate", "ratio");
    for q in [0.5, 0.9, 0.99, 0.999] {
        let truth = values[(q * (N - 1) as f64) as usize];
        let estimate = kll.quantile(q).unwrap();
        println!(
            "{:<10.3} {:>14.3} {:>13.4}",
            q,
            estimate.0,
            estimate.0 / truth
        );
    }
    println!("KLL's guarantee is about rank, not value: at p99.9 the relative");
    println!("value error is whatever the tail shape makes it.\n");

    // -- Part 3: exact merge ----------------------------------------------------------
    println!("Part 3: merged halves vs. whole stream");
    let mut left = DdSketch::new(0.02);
    let mut right = DdSketch::new(0.02);
    let mut whole = DdSketch::new(0.02);
    for (i, &v) in values.iter().enumerate() {
        if i % 2 == 0 {
            left.insert_item(&v);
        } else {
            right.insert_item(&v);
        }
        whole.insert_item(&v);
    }
    left.merge_from(&right).unwrap();
    let identical = left.buckets() == whole.buckets();
    println!(
        "merged halves identical to whole stream (buckets): {identical}; p99: {:.3} vs {:.3}",
        left.quantile(0.99).unwrap(),
        whole.quantile(0.99).unwrap()
    );
}
