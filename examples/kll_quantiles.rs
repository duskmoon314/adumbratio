//! Reproduces the rank-error behavior of the KLL quantile sketch.
//!
//! # What the paper evaluates
//!
//! - Zohar Karnin, Kevin Lang, and Edo Liberty, "Optimal Quantile
//!   Approximation in Streams", FOCS 2016.
//!   The paper's promise: from a stream of `n` values, answer quantile
//!   queries with *rank* error bounded independently of the distribution —
//!   the estimated `q`-quantile's true rank lies in `q +/- eps` — while
//!   storing about `k + log2(n)` values. Their evaluation measures the
//!   observed rank error across quantiles and stream distributions
//!   (uniform, normal, and skewed), and shows it tracking `~1/k` with no
//!   dependence on `n`. That invariance to the input distribution is what
//!   makes compactor sketches the production standard (Apache
//!   DataSketches, BigQuery).
//!
//! # What this example does
//!
//! 1. Streams one million values from two very different distributions
//!    (uniform and Zipf-skewed) into `k = 200` sketches and prints the
//!    worst absolute rank error across all percentiles — the number the
//!    paper says should sit around `1/k`, on both distributions.
//! 2. Prints the same measurement at several capacities `k`, showing the
//!    error shrinking like `1/k` while memory stays under a kilobyte of
//!    stored values.
//! 3. Demonstrates mergeability: two half-streams sketched separately and
//!    merged answer as accurately as one sketch over the whole stream.
//!
//! Streams are driven by a seeded xorshift, so the output is reproducible.
//!
//! Run with: `cargo run --release --example kll_quantiles`

use adumbratio::policy::{RngLite, XorShift64};
use adumbratio::sketch::KllSketch;
use adumbratio::traits::Merge;

const N: u64 = 1_000_000;

/// Worst absolute rank error across all percentiles. The rank of a value
/// with duplicates is an interval [fraction < v, fraction <= v]; the error
/// of answering v for quantile q is the distance from q to that interval
/// (zero when q falls inside it). This is the standard rank-error
/// definition and matters exactly on skewed, duplicate-heavy streams.
fn rank_error(sorted_truth: &[u64], q: f64, estimate: u64) -> f64 {
    let n = sorted_truth.len() as f64;
    let first = sorted_truth.partition_point(|&v| v < estimate) as f64 / n;
    let last = sorted_truth.partition_point(|&v| v <= estimate) as f64 / n;
    if q < first {
        first - q
    } else if q > last {
        q - last
    } else {
        0.0
    }
}

fn worst_rank_error(sketch: &KllSketch<u64>, sorted_truth: &[u64]) -> f64 {
    let mut worst = 0.0_f64;
    for qi in 1..100 {
        let q = qi as f64 / 100.0;
        let estimate = sketch.quantile(q).unwrap();
        worst = worst.max(rank_error(sorted_truth, q, estimate));
    }
    worst
}

fn main() {
    println!("adumbratio — KLL rank-error behavior (Karnin–Lang–Liberty 2016)");
    println!("n = {N} values per stream, per-level capacity k\n");

    // -- Part 1: distribution invariance ---------------------------------------
    // Uniform values over [0, 2^32) and a Zipf-like skewed distribution. The
    // paper's point is that the rank error does not care which one is used.
    println!("Part 1: worst rank error across percentiles, k = 200");
    println!("{:<16} {:>18}", "distribution", "worst |rank error|");

    let mut uniform = Vec::with_capacity(N as usize);
    let mut sketch = KllSketch::with_seed(200, 1);
    let mut rng = XorShift64::new(11);
    for _ in 0..N {
        let v = rng.next_u64() % (1 << 32);
        sketch.insert_item(&v);
        uniform.push(v);
    }
    uniform.sort_unstable();
    println!("{:<16} {:>18.4}", "uniform", worst_rank_error(&sketch, &uniform));

    // Zipf-like skew: values proportional to 1/rank of the stream order.
    let mut skewed = Vec::with_capacity(N as usize);
    let mut sketch = KllSketch::with_seed(200, 2);
    let mut rng = XorShift64::new(13);
    for _ in 0..N {
        // Power-law-ish values via inverse transform: u^(-1/1.5) - 1.
        let u = (rng.next_u64() as f64 + 1.0) / u64::MAX as f64;
        let v = (u.powf(-1.0 / 1.5) - 1.0) as u64;
        sketch.insert_item(&v);
        skewed.push(v);
    }
    skewed.sort_unstable();
    println!("{:<16} {:>18.4}", "zipf-like", worst_rank_error(&sketch, &skewed));
    println!("1/k = {:.4}; the paper's claim is distribution independence.\n", 1.0 / 200.0);

    // -- Part 2: error vs. capacity --------------------------------------------
    println!("Part 2: error and memory as k grows (uniform stream)");
    println!("{:<8} {:>16} {:>16}", "k", "worst |rank error|", "values stored");
    for k in [50_usize, 100, 200, 400] {
        let mut sketch = KllSketch::with_seed(k, 1);
        let mut rng = XorShift64::new(11);
        for _ in 0..N {
            sketch.insert_item(&(rng.next_u64() % (1 << 32)));
        }
        println!(
            "{:<8} {:>16.4} {:>16}",
            k,
            worst_rank_error(&sketch, &uniform),
            sketch.storage_bytes() / size_of::<u64>()
        );
    }

    // -- Part 3: merge equivalence ----------------------------------------------
    // Two half-streams sketched and merged must answer like one sketch over
    // the whole stream — the property that makes KLL usable in distributed
    // systems.
    println!("\nPart 3: merged halves vs. whole stream");
    let mut left = KllSketch::with_seed(200, 21);
    let mut right = KllSketch::with_seed(200, 22);
    let mut whole = KllSketch::with_seed(200, 23);
    let mut rng = XorShift64::new(17);
    for i in 0..N {
        let v = rng.next_u64() % (1 << 32);
        if i % 2 == 0 {
            left.insert_item(&v);
        } else {
            right.insert_item(&v);
        }
        whole.insert_item(&v);
    }
    left.merge_from(&right).unwrap();
    println!(
        "worst rank error: merged = {:.4}, whole-stream = {:.4}",
        worst_rank_error(&left, &uniform),
        worst_rank_error(&whole, &uniform)
    );
}
