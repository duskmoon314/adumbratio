//! Demonstrates the theta sketch's set algebra against known set sizes.
//!
//! # What the paper establishes
//!
//! - Ziv Bar-Yossef, T. S. Jayram, Ravi Kumar, D. Sivakumar, and Luca
//!   Trevisan, "Counting Distinct Elements in a Data Stream", RANDOM 2002.
//!   <https://doi.org/10.1007/3-540-45726-7_1>
//!   The bottom-k (KMV) estimator: keep the `k` smallest hash values of a
//!   stream; with threshold `theta` (the k-th smallest), the distinct count
//!   is `(k-1)/theta` with standard error about `1/sqrt(k)`. Because the
//!   sample is defined by a *threshold* rather than fixed registers, two
//!   sketches combine under all Boolean set operations — union by keeping
//!   the k smallest of both lists, intersection and difference by filtering
//!   common or distinct values at the shared threshold. That is the
//!   advantage over HyperLogLog, which supports only union.
//!
//! # What this example does
//!
//! 1. Builds two sets with a known overlap and prints the theta estimates
//!    for union, intersection, difference, and Jaccard similarity next to
//!    the exact answers — the operations HyperLogLog cannot express.
//! 2. Shows the error scaling with `k`: doubling the retained hashes cuts
//!    the error by about 30% (the `1/sqrt(k)` law).
//! 3. Notes the exact mode: below `k` distinct items the sketch answers
//!    with zero error.
//!
//! All sets are sequential integer ranges, so the output is reproducible.
//!
//! Run with: `cargo run --release --example theta_set_ops`

use adumbratio::sketch::ThetaSketch;

/// Builds sets A and B of size `size` sharing `overlap` items.
fn pair(k: usize, size: u64, overlap: u64, seed: u64) -> (ThetaSketch, ThetaSketch) {
    let mut a = ThetaSketch::with_seed(k, seed);
    let mut b = ThetaSketch::with_seed(k, seed);
    for i in 0..(size - overlap) {
        a.insert_item(&i);
        b.insert_item(&(1_000_000 + i));
    }
    for i in 0..overlap {
        a.insert_item(&(2_000_000 + i));
        b.insert_item(&(2_000_000 + i));
    }
    (a, b)
}

fn main() {
    println!("adumbratio — theta sketch set algebra (Bar-Yossef et al. 2002)\n");

    // -- Part 1: all four set operations ----------------------------------------
    // Each estimate is unbiased but noisy; means over five seeds sit on the
    // truth (the paper's point), which is what we print.
    let (size, overlap) = (50_000_u64, 12_500_u64);
    let (union_t, inter_t, diff_t) = (
        2.0 * size as f64 - overlap as f64,
        overlap as f64,
        size as f64 - overlap as f64,
    );
    println!("Part 1: mean estimates vs. truth (|A| = |B| = {size}, |A∩B| = {overlap}, k = 512, 5 seeds)");
    println!(
        "{:<14} {:>14} {:>14} {:>12}",
        "operation", "mean estimate", "truth", "rel. error"
    );
    let mut means = [0.0_f64; 4];
    for seed in 1..=5_u64 {
        let (a, b) = pair(512, size, overlap, seed);
        means[0] += a.estimate_union(&b) / 5.0;
        means[1] += a.estimate_intersection(&b) / 5.0;
        means[2] += a.estimate_difference(&b) / 5.0;
        means[3] += a.jaccard(&b) / 5.0;
    }
    let rows = [
        ("union", means[0], union_t),
        ("intersection", means[1], inter_t),
        ("difference", means[2], diff_t),
        ("jaccard", means[3], overlap as f64 / union_t),
    ];
    for (name, estimate, truth) in rows {
        let rel = (estimate - truth).abs() / truth;
        println!("{:<14} {:>14.1} {:>14.1} {:>11.2}%", name, estimate, truth, rel * 100.0);
    }
    let (a, b) = pair(512, size, overlap, 7);
    let _ = b;
    println!("standard error 1/sqrt(k) = {:.2}%\n", a.standard_error() * 100.0);

    // -- Part 2: error vs. k ------------------------------------------------------
    println!("Part 2: mean |union error| as k doubles (3 seeds)");
    println!("{:<10} {:>16} {:>16}", "k", "mean |rel. error|", "1/sqrt(k)");
    for k in [128_usize, 256, 512, 1024] {
        let mean_error: f64 = (1..=3_u64)
            .map(|seed| {
                let (a, b) = pair(k, size, overlap, seed);
                (a.estimate_union(&b) - union_t).abs() / union_t
            })
            .sum::<f64>()
            / 3.0;
        println!(
            "{:<10} {:>15.2}% {:>15.2}%",
            k,
            mean_error * 100.0,
            100.0 / (k as f64).sqrt()
        );
    }

    // -- Part 3: exact mode ---------------------------------------------------------
    println!("\nPart 3: below k distinct items the sketch is exact");
    let mut small = ThetaSketch::new(128);
    for i in 0..100_u64 {
        small.insert_item(&i);
    }
    println!(
        "100 items in a k = 128 sketch: cardinality = {} (exact = 100)",
        small.cardinality()
    );
}
