//! Demonstrates SimHash cosine estimation and exact linear merging.
//!
//! # What the papers establish
//!
//! - Moses S. Charikar, "Similarity Estimation Techniques from Rounding
//!   Algorithms", STOC 2002. <https://doi.org/10.1145/509907.509965>
//!   The rounding trick: sign patterns of random projections preserve
//!   angles, so a small bit signature estimates cosine similarity by
//!   Hamming distance. Applied to item multisets (0/1 incidence vectors),
//!   the angle between two sets' vectors is recovered from `pi * h / f`
//!   where `h` is the signature distance.
//!
//! - Gurmeet Singh Manku, Arvind Jain, and Anish Das Sarma, "Detecting
//!   Near-Duplicates for Web Crawling", WWW 2007.
//!   <https://doi.org/10.1145/1242572.1242592>
//!   The deployment story: SimHash is *linear*, so the sketch of a union
//!   is the element-wise sum of the sketches — the property Google used
//!   for near-duplicate detection over billions of pages. Part 2 below
//!   shows the merge being bit-exact, not approximate.
//!
//! # What this example does
//!
//! 1. Builds equal-size sets with controlled overlaps and prints the true
//!    incidence cosine next to the SimHash estimate — honest about the
//!    64-bit signature's coarseness.
//! 2. Shows linear merging is exact: two half-streams sketched separately
//!    and merged produce the identical signature of one sketch over the
//!    whole stream.
//!
//! All sets are sequential integer ranges, so the output is reproducible.
//!
//! Run with: `cargo run --release --example simhash_cosine`

use adumbratio::sketch::SimHash;
use adumbratio::traits::Merge;

fn main() {
    println!("adumbratio — SimHash cosine similarity (Charikar 2002)\n");

    // -- Part 1: estimate vs. true cosine -----------------------------------------
    // Equal-size sets sharing c items have incidence cosine c / s. Each row
    // averages three seeds: single 64-bit estimates scatter, means track.
    println!("Part 1: estimated vs. true cosine (s = 20000 items per set, 3 seeds)");
    println!(
        "{:<12} {:>14} {:>14} {:>14}",
        "true cosine", "mean estimate", "mean |error|", "noise bound"
    );
    let s = 20_000_u64;
    for (c, truth) in [
        (0_u64, 0.0_f64),
        (5_000, 0.25),
        (10_000, 0.5),
        (15_000, 0.75),
        (20_000, 1.0),
    ] {
        let mut mean = 0.0_f64;
        let mut mean_error = 0.0_f64;
        for seed in 1..=3_u64 {
            let mut a = SimHash::with_seed(seed);
            let mut b = SimHash::with_seed(seed);
            for i in 0..(s - c) {
                a.insert_item(&i);
                b.insert_item(&(1_000_000 + i));
            }
            for i in 0..c {
                a.insert_item(&(2_000_000 + i));
                b.insert_item(&(2_000_000 + i));
            }
            let estimate = a.estimated_cosine(&b);
            mean += estimate / 3.0;
            mean_error += (estimate - truth).abs() / 3.0;
        }
        println!(
            "{:<12.2} {:>14.3} {:>14.3} {:>14.2}",
            truth, mean, mean_error, 0.2
        );
    }

    // -- Part 2: exact linear merging ---------------------------------------------
    println!("\nPart 2: merge is exact (the Manku et al. deployment property)");
    let mut left = SimHash::with_seed(11);
    let mut right = SimHash::with_seed(11);
    let mut single = SimHash::with_seed(11);
    for i in 0..50_000_u64 {
        if i % 2 == 0 {
            left.insert_item(&i);
        } else {
            right.insert_item(&i);
        }
        single.insert_item(&i);
    }
    left.merge_from(&right).unwrap();
    println!(
        "merged halves vs. whole stream: signature distance = {} bits (exact = 0)",
        left.hamming_distance(&single)
    );
    println!("\nTakeaway: coarse per-query estimates, but a mergeable,");
    println!("index-friendly 64-bit fingerprint — the near-dup trade-off.");
}
