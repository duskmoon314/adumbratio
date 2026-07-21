//! Reproduces the point-query error experiments of the frequency sketches.
//!
//! # What the papers evaluate
//!
//! - Graham Cormode and S. Muthukrishnan, "An Improved Data Stream Summary:
//!   The Count-Min Sketch and its Applications", Journal of Algorithms 2005.
//!   <https://doi.org/10.1016/j.jalgor.2003.12.001>
//!   The guarantee: with `w = ceil(e/eps)` and `d = ceil(ln(1/delta))`, the
//!   point estimate never underestimates and overshoots by at most
//!   `eps * N` with probability `>= 1 - delta`, where `N` is the stream
//!   length. The paper's experiments (Section 6) use synthetic Zipf streams,
//!   exactly what we generate here — their argument being that real traffic
//!   is heavy-tailed, so a small set of heavy hitters dominates.
//!
//! - Cristian Estan and George Varghese, "New Directions in Traffic
//!   Measurement and Accounting", SIGCOMM 2002.
//!   <https://doi.org/10.1145/633025.633056>
//!   Proposes *conservative update*: on insert, increment only the counters
//!   equal to the row minimum. The query path and the error guarantee are
//!   unchanged, but measured overestimation drops substantially on skewed
//!   traffic. The comparison below should show a clear reduction in average
//!   error at zero extra space cost.
//!
//! - Moses Charikar, Kevin Chen, and Martin Farach-Colton, "Finding
//!   Frequent Items in Data Streams", TCS 2004.
//!   <https://doi.org/10.1016/j.tcs.2003.10.024>
//!   Count Sketch: same row layout, but a random sign per row makes
//!   collision noise unbiased, and a median read replaces the min. Its
//!   error is bounded against the L2 norm of the stream, so it shines on
//!   heavy hitters; we include it in the heavy-hitter table.
//!
//! # What this example does
//!
//! 1. Generates a Zipf(1.1) stream — the skew the papers assume — with known
//!    true frequencies.
//! 2. Builds plain Count-Min, conservative-update Count-Min, and Count
//!    Sketch, and measures, over every distinct item: underestimates (must
//!    be zero for Count-Min), mean/max error, and the fraction of items
//!    violating the `eps * N` bound (must stay near zero, `<= delta`).
//! 3. Varies the width to show the error shrinking like `1/w`, the paper's
//!    central trade-off.
//! 4. Prints the top-10 heavy hitters with all three estimates side by side.
//!
//! The stream is driven by a seeded xorshift, so the output is
//! reproducible. Run with: `cargo run --release --example count_min_zipf`

use adumbratio::policy::{RngLite, XorShift64};
use adumbratio::sketch::{CountMinSketch, CountSketch};

/// Stream length `N`: the error bound is `eps * N`.
const EVENTS: usize = 200_000;

/// Distinct items in the stream.
const UNIVERSE: u64 = 10_000;

/// Zipf exponent 1.1, the skew used in the Count-Min paper's experiments.
const ZIPF_S: f64 = 1.1;

fn main() {
    println!("adumbratio — frequency-sketch error on a Zipf({ZIPF_S}) stream");
    println!("N = {EVENTS} events over {UNIVERSE} distinct items\n");

    let counts = zipf_counts(UNIVERSE, EVENTS, ZIPF_S, /* seed */ 11);

    // -- Part 1: error metrics, plain vs. conservative -----------------------
    let (epsilon, delta) = (0.001, 0.01);
    let eps_n = (epsilon * EVENTS as f64) as u64;
    println!(
        "Part 1: Count-Min with eps = {epsilon}, delta = {delta} (w = {}, d = {}), bound eps*N = {eps_n}",
        CountMinSketch::with_error(epsilon, delta).geometry().width,
        CountMinSketch::with_error(epsilon, delta).geometry().depth,
    );
    println!(
        "{:<16} {:>14} {:>12} {:>12} {:>16}",
        "variant", "underestimates", "mean error", "max error", "violations>epsN"
    );

    let mut plain = CountMinSketch::with_error_and_seed(epsilon, delta, 5);
    let mut conservative = CountMinSketch::conservative_with_error_and_seed(epsilon, delta, 5);
    for (item, &count) in counts.iter().enumerate() {
        for _ in 0..count {
            plain.insert_item(&(item as u64));
            conservative.insert_item(&(item as u64));
        }
    }
    report("plain", &counts, eps_n, |i| plain.estimate_item(&i));
    report("conservative", &counts, eps_n, |i| {
        conservative.estimate_item(&i)
    });

    // -- Part 2: error scaling with width ------------------------------------
    // The bound is eps*N, so halving eps (doubling w) should about halve the
    // observed average error: the space/accuracy trade-off made visible.
    println!("\nPart 2: mean error as the width grows (plain update)");
    println!("{:<10} {:>10} {:>12} {:>12}", "eps", "width w", "mean error", "bound eps*N");
    for eps in [0.01_f64, 0.004, 0.002, 0.001] {
        let mut sketch = CountMinSketch::with_error_and_seed(eps, 0.01, 5);
        for (item, &count) in counts.iter().enumerate() {
            for _ in 0..count {
                sketch.insert_item(&(item as u64));
            }
        }
        let total_error: u64 = counts
            .iter()
            .enumerate()
            .map(|(i, &truth)| sketch.estimate_item(&(i as u64)) - truth)
            .sum();
        println!(
            "{:<10} {:>10} {:>12.1} {:>12.0}",
            eps,
            sketch.geometry().width,
            total_error as f64 / counts.len() as f64,
            eps * EVENTS as f64
        );
    }

    // -- Part 3: heavy hitters ------------------------------------------------
    // Where Count Sketch earns its L2 bound: heavy items dominate the
    // stream, so their relative error is tiny for every variant.
    let mut count_sketch = CountSketch::with_error_and_seed(0.02, 0.01, 5);
    for (item, &count) in counts.iter().enumerate() {
        for _ in 0..count {
            count_sketch.insert_item(&(item as u64));
        }
    }

    println!("\nPart 3: top-10 heavy hitters (true vs. estimates)");
    println!(
        "{:<10} {:>12} {:>14} {:>14} {:>14}",
        "item", "true", "CM plain", "CM conservative", "Count Sketch"
    );
    let mut order: Vec<usize> = (0..counts.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(counts[i]));
    for &i in order.iter().take(10) {
        let item = i as u64;
        println!(
            "{:<10} {:>12} {:>14} {:>14} {:>14}",
            item,
            counts[i],
            plain.estimate_item(&item),
            conservative.estimate_item(&item),
            count_sketch.estimate_item(&item),
        );
    }
    println!("\nExpected: zero underestimates for Count-Min, conservative update");
    println!("clearly below plain on mean error, and heavy hitters recovered almost");
    println!("exactly by every variant — the papers' claims in three tables.");
}

/// Prints one row of error metrics for a Count-Min variant.
fn report(name: &str, counts: &[u64], eps_n: u64, estimate: impl Fn(u64) -> u64) {
    let mut underestimates = 0_usize;
    let mut total_error = 0_u64;
    let mut max_error = 0_u64;
    let mut violations = 0_usize;
    for (i, &truth) in counts.iter().enumerate() {
        let est = estimate(i as u64);
        if est < truth {
            underestimates += 1;
            continue;
        }
        let error = est - truth;
        total_error += error;
        max_error = max_error.max(error);
        if error > eps_n {
            violations += 1;
        }
    }
    println!(
        "{:<16} {:>14} {:>12.1} {:>12} {:>16}",
        name,
        underestimates,
        total_error as f64 / counts.len() as f64,
        max_error,
        violations
    );
}

/// Simulates a Zipf(s) stream and returns true per-item frequencies.
///
/// Items are drawn by inverse-CDF sampling over the harmonic-like weights
/// `1 / (i+1)^s`, the same construction used for the synthetic workloads in
/// the Count-Min paper's evaluation section.
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
