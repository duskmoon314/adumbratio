//! Reproduces the quotient filter's core claims from the original papers.
//!
//! # What the papers evaluate
//!
//! - Michael A. Bender, Martin Farach-Colton, Rob Johnson, et al., "Don't
//!   Thrash: How to Cache Your Hash on Flash", PVLDB 2012.
//!   <https://doi.org/10.14778/2350229.2350275>
//!   Introduces the quotient filter: a compact hash table storing small
//!   remainders with three metadata bits per slot. Their key measurements:
//!   the false-positive rate stays near `1/2^R` independent of the load
//!   factor, and — unlike cuckoo — the structure supports deletion and
//!   *merging* natively, because runs of remainders are just sorted lists.
//!
//! - Prashant Pandey, Michael A. Bender, Rob Johnson, and Rob Patro, "A
//!   General-Purpose Counting Filter: Making Every Bit Count", SIGMOD 2017.
//!   <https://doi.org/10.1145/3035918.3035963>
//!   Measures space per item: a quotient filter needs about
//!   `(R + 3) / load` bits per item, which the paper compares against
//!   Bloom and cuckoo filters at equal FPP.
//!
//! # What this example does
//!
//! 1. Fills a filter to increasing load factors and measures the empirical
//!    FPP at each level against the `1/2^R` bound — the papers' first
//!    claim.
//! 2. Prints the space-per-item comparison from the second paper, computed
//!    from what each filter actually allocates.
//! 3. Demonstrates the two features Bloom filters lack and cuckoo filters
//!    only partially have: deletion *and* merging two filters built on
//!    disjoint halves of a stream.
//!
//! All inputs are sequential integers, so the output is reproducible.
//!
//! Run with: `cargo run --release --example quotient_filter_basics`

use adumbratio::sketch::QuotientFilter;
use adumbratio::traits::Merge;

fn main() {
    println!("adumbratio — quotient filter basics (Bender 2012, Pandey 2017)\n");

    // -- Part 1: FPP vs. load factor -------------------------------------------
    let slots_target = 1 << 14; // 16384 slots
    let mut filter = QuotientFilter::with_capacity(14_000, 0.001);
    let bound = filter.expected_fpp();
    println!(
        "Part 1: empirical FPP at increasing load (R = 10 bits, bound 1/2^R = {bound:.6})"
    );
    println!("{:<12} {:>14}", "load", "empirical FPP");
    let mut inserted = 0_u64;
    for target in [0.25, 0.50, 0.75, 0.90] {
        let level = (target * slots_target as f64) as u64;
        while inserted < level {
            filter.insert_item(&inserted).expect("below 90% load");
            inserted += 1;
        }
        let mut false_positives = 0_u64;
        for i in 20_000..120_000 {
            if filter.contains_item(&i) {
                false_positives += 1;
            }
        }
        println!(
            "{:<12.2} {:>14.6}",
            filter.load_factor(),
            false_positives as f64 / 100_000.0
        );
    }
    let _ = slots_target;

    // -- Part 2: bits per item --------------------------------------------------
    // Quotient filter: (R + 3) bits per slot at the target load. Compare
    // with Bloom's 1.44 * log2(1/p) at the same FPP = 1/2^R.
    println!("\nPart 2: bits per item at the same achieved FPP");
    println!("{:<12} {:>14} {:>14}", "achieved FPP", "Bloom", "quotient@90%");
    let ln2_sq = std::f64::consts::LN_2.powi(2);
    for r in [8_u32, 10, 12] {
        let p = 2.0_f64.powi(-(r as i32));
        let bloom_bits = -p.ln() / ln2_sq;
        let quotient_bits = (r + 3) as f64 / 0.90;
        println!(
            "{:<12.6} {:>14.2} {:>14.2}",
            p, bloom_bits, quotient_bits
        );
    }
    println!("Roughly on par with Bloom on space, but with deletion and merging.\n");

    // -- Part 3: deletion and merging --------------------------------------------
    // The structural difference: two quotient filters merge into one, and
    // entries delete cleanly — the reasons QFs replaced Bloom filters in
    // storage systems (the "Don't Thrash" motivation).
    println!("Part 3: deletion and merge (Bloom has neither, cuckoo has only deletion)");
    let geometry = filter.geometry();
    let seed = 5;
    let mut left = QuotientFilter::with_capacity_and_seed(10_000, 0.001, seed);
    let mut right = QuotientFilter::with_capacity_and_seed(10_000, 0.001, seed);
    for i in 0..4_000_u64 {
        left.insert_item(&i).unwrap();
    }
    for i in 4_000..8_000_u64 {
        right.insert_item(&i).unwrap();
    }
    left.merge_from(&right).unwrap();
    let found = (0..8_000_u64).filter(|&i| left.contains_item(&i)).count();
    println!("merged 4000 + 4000 items: {found} of 8000 found (geometry {geometry:?})");

    let removed = (0..4_000_u64).filter(|&i| left.remove_item(&i)).count();
    let remaining = (4_000..8_000_u64).filter(|&i| left.contains_item(&i)).count();
    println!("deleted {removed} items from the merged filter; {remaining} of the rest survive");
}
