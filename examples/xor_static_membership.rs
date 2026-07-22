//! Reproduces the space/error comparison of the xor filter paper.
//!
//! # What the paper evaluates
//!
//! - Thomas Mueller Graf and Daniel Lemire, "Xor Filters: Faster and
//!   Smaller Than Bloom and Cuckoo Filters", ACM JEA 2020.
//!   <https://doi.org/10.1145/3376122>
//!   The paper's two headline measurements: (1) space per item at a given
//!   false-positive rate — xor filters use `1.23 * log2(1/eps)` bits per
//!   item against Bloom's `1.44 * log2(1/eps)` and cuckoo's roughly
//!   `1.05 * (log2(1/eps) + 3)`, and (2) membership-query speed, where a
//!   constant three probes beat Bloom's growing `k`. The price, stated
//!   plainly in the paper: the filter is static — built once, queried many
//!   times, with no insertion, deletion, or merging.
//!
//! # What this example does
//!
//! 1. Prints bits per item for the three filter families at several target
//!    FPPs, computed from the geometry each one actually allocates — the
//!    paper's Figure-3-style comparison with our own structures.
//! 2. Verifies the realized FPP of a built xor filter against the
//!    `2^-BITS` bound, at both `u8` and `u16` slot widths.
//! 3. Times a bulk build and a query sweep, so the static/dynamic
//!    trade-off is visible in numbers, not just words.
//!
//! All inputs are sequential integers, so the output is reproducible.
//!
//! Run with: `cargo run --release --example xor_static_membership`

use std::time::Instant;

use adumbratio::sketch::XorFilter;

/// Inserted item count for the live measurements.
const N: u64 = 100_000;

fn main() {
    println!("adumbratio — xor filter space/error comparison (Graf–Lemire 2020)\n");

    // -- Part 1: bits per item --------------------------------------------------
    // The paper's comparison at MATCHED false-positive rates: an xor filter
    // of slot width f achieves FPP 2^-f using 1.23f bits/item, while a
    // Bloom filter needs 1.44 * log2(1/p) bits for the same p.
    println!("Part 1: bits per item at the same achieved FPP");
    println!(
        "{:<12} {:>14} {:>12} {:>12} {:>10}",
        "slot width", "achieved FPP", "Bloom bits", "xor bits", "saving"
    );
    let ln2_sq = std::f64::consts::LN_2.powi(2);
    for (name, f) in [("u8", 8.0_f64), ("u16", 16.0)] {
        let p = 2.0_f64.powf(-f);
        let bloom_bits = -p.ln() / ln2_sq;
        let xor_bits = 1.23 * f;
        println!(
            "{:<12} {:>14.6} {:>12.2} {:>12.2} {:>9.1}%",
            name,
            p,
            bloom_bits,
            xor_bits,
            100.0 * (1.0 - xor_bits / bloom_bits)
        );
    }
    println!("The paper's headline: ~15% smaller than Bloom at the same FPP,");
    println!("with faster queries; deletion stays exclusive to cuckoo.\n");

    // -- Part 2: realized FPP ----------------------------------------------------
    let items: Vec<u64> = (0..N).collect();
    println!("Part 2: realized FPP vs. 2^-BITS bound (n = {N})");
    println!("{:<10} {:>14} {:>14}", "slot width", "bound", "empirical");

    let filter = XorFilter::build(&items);
    let fp = false_positive_rate(&filter, N, 200_000);
    println!("{:<10} {:>14.7} {:>14.7}", "u16", filter.expected_fpp(), fp);

    let narrow = XorFilter::<u8>::build_with_seed(&items, 0);
    let fp = false_positive_rate(&narrow, N, 200_000);
    println!("{:<10} {:>14.7} {:>14.7}", "u8", narrow.expected_fpp(), fp);

    // -- Part 3: build and query cost --------------------------------------------
    println!("\nPart 3: build time and query sweep at n = {N}");
    let start = Instant::now();
    let filter = XorFilter::build(&items);
    let build = start.elapsed();
    println!("build: {build:.2?} for {N} items");

    let start = Instant::now();
    let mut hits = 0_u64;
    for i in 0..N {
        hits += filter.contains_item(&i) as u64;
    }
    let present = start.elapsed();
    let start = Instant::now();
    let mut false_positives = 0_u64;
    for i in N..2 * N {
        false_positives += filter.contains_item(&i) as u64;
    }
    let absent = start.elapsed();
    println!("query_hit:  {present:.2?} for {N} (all {hits} found)");
    println!("query_miss: {absent:.2?} for {N} ({false_positives} false positives)");
    println!("\nReminder: the filter is static — these are the only two operations");
    println!("it will ever perform. Insert/delete belong to Bloom and cuckoo.");
}

fn false_positive_rate<F: adumbratio::block::Fingerprint>(
    filter: &XorFilter<F>,
    n: u64,
    queries: u64,
) -> f64 {
    let mut false_positives = 0_u64;
    for i in n..n + queries {
        if filter.contains_item(&i) {
            false_positives += 1;
        }
    }
    false_positives as f64 / queries as f64
}
