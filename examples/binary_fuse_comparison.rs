//! Compares the binary fuse filter against xor and Bloom on space and FPP.
//!
//! # What the paper evaluates
//!
//! - Thomas Mueller Graf and Daniel Lemire, "Binary Fuse Filters: Fast and
//!   Smaller Than Xor Filters", ACM JEA 2022. <https://doi.org/10.1145/3510449>
//!   The paper's two claims, measured against xor filters and Bloom
//!   filters: (1) space — binary fuse uses about `1.125 * f` bits per item
//!   for `f`-bit fingerprints against xor's `1.23 * f` and Bloom's
//!   `1.44 * log2(1/eps)`, and (2) build and query speed of the same
//!   order. The construction's key trick, faithfully reproduced here: the
//!   first position spreads over the whole table while the other two are
//!   xor-masked inside power-of-two segments, letting the hypergraph peel
//!   at a higher load factor.
//!
//! # What this example does
//!
//! 1. Prints bits per item for the three filter families at matched
//!    achieved FPP — the paper's headline comparison, computed from the
//!    tables this crate actually allocates (including binary fuse's
//!    empirical size factor).
//! 2. Verifies the realized FPP against the `2^-BITS` bound at u8 and u16.
//! 3. Times a 100k-item build and query sweep against the xor filter built
//!    on the same items.
//!
//! All inputs are sequential integers, so the output is reproducible.
//!
//! Run with: `cargo run --release --example binary_fuse_comparison`

use std::time::Instant;

use adumbratio::sketch::{BinaryFuseFilter, XorFilter};

const N: u64 = 100_000;

fn main() {
    println!("adumbratio — binary fuse vs. xor and Bloom (Graf–Lemire 2022)\n");

    let items: Vec<u64> = (0..N).collect();
    let fuse = BinaryFuseFilter::build(&items);
    let xor = XorFilter::build(&items);

    // -- Part 1: allocated space ------------------------------------------------
    println!("Part 1: allocated bits per item (u16 fingerprints, n = {N})");
    println!(
        "{:<16} {:>14} {:>16}",
        "filter", "table slots", "bits per item"
    );
    println!(
        "{:<16} {:>14} {:>16.2}",
        "xor", xor.table_len(),
        16.0 * xor.table_len() as f64 / N as f64
    );
    println!(
        "{:<16} {:>14} {:>16.2}",
        "binary fuse",
        fuse.table_len(),
        16.0 * fuse.table_len() as f64 / N as f64
    );
    let bloom_bits = 1.44 * (1.0 / fuse.expected_fpp()).log2();
    println!("{:<16} {:>14} {:>16.2}", "Bloom (same FPP)", "-", bloom_bits);
    println!("paper's claim: ~1.125f vs ~1.23f bits/item — fuse wins.\n");

    // -- Part 2: realized FPP -----------------------------------------------------
    println!("Part 2: realized FPP vs. 2^-BITS bound");
    println!("{:<10} {:>14} {:>14}", "slot width", "bound", "empirical");
    let fp = false_positive_rate_fuse(&fuse, N, 200_000);
    println!("{:<10} {:>14.7} {:>14.7}", "u16", fuse.expected_fpp(), fp);
    let narrow = BinaryFuseFilter::<u8>::build_with_seed(&items, 0);
    let fp = false_positive_rate_fuse(&narrow, N, 200_000);
    println!("{:<10} {:>14.7} {:>14.7}", "u8", narrow.expected_fpp(), fp);

    // -- Part 3: build and query speed ---------------------------------------------
    println!("\nPart 3: build time and query sweep at n = {N}");
    let start = Instant::now();
    let fuse = BinaryFuseFilter::build(&items);
    let fuse_build = start.elapsed();
    let start = Instant::now();
    let xor = XorFilter::build(&items);
    let xor_build = start.elapsed();
    println!("build: fuse {fuse_build:.2?}, xor {xor_build:.2?}");

    let start = Instant::now();
    let mut hits = 0_u64;
    for i in 0..N {
        hits += fuse.contains_item(&i) as u64;
    }
    let fuse_query = start.elapsed();
    let start = Instant::now();
    let mut hits2 = 0_u64;
    for i in 0..N {
        hits2 += xor.contains_item(&i) as u64;
    }
    let xor_query = start.elapsed();
    println!("query_hit: fuse {fuse_query:.2?} ({hits} found), xor {xor_query:.2?} ({hits2} found)");
}

fn false_positive_rate_fuse<F: adumbratio::block::Fingerprint>(
    filter: &BinaryFuseFilter<F>,
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
