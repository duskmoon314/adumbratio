//! Reproduces the semi-sorted cuckoo space optimization from the paper.
//!
//! # What the paper evaluates
//!
//! - Bin Fan, Dave G. Andersen, Michael Kaminsky, and Michael D.
//!   Mitzenmacher, "Cuckoo Filter: Practically Better Than Bloom", CoNEXT
//!   2014, Section 4.2. <https://doi.org/10.1145/2674005.2674994>
//!   The paper's own space optimization: the four fingerprints of a bucket
//!   are stored *sorted*. A sorted multiset of four `f`-bit values carries
//!   about `4f - 4.6` bits of information instead of `4f`, so one bit per
//!   fingerprint is saved at no cost to the false-positive rate — in their
//!   words, "semi-sorting saves one bit per fingerprint". The price is
//!   construction and lookup complexity: buckets are encoded and decoded
//!   rather than read directly. Our implementation packs the sorted bucket
//!   as a combinatorial rank, 56 bits instead of 64.
//!
//! # What this example does
//!
//! 1. Prints allocated bits per item for the plain u16 cuckoo filter and
//!    the semi-sorted variant at the same false-positive rate — the
//!    paper's one-bit saving made concrete.
//! 2. Verifies the realized FPP against the shared `8/2^15` bound.
//! 3. Times insert and query on both filters, so the speed-for-space
//!    trade is visible in numbers.
//!
//! All inputs are sequential integers, so the output is reproducible.
//!
//! Run with: `cargo run --release --example semi_sorted_cuckoo`

use std::time::Instant;

use adumbratio::sketch::{CuckooFilter, SemiSortedCuckooFilter};

const N: u64 = 10_000;

fn main() {
    println!("adumbratio — semi-sorted cuckoo (Fan et al. 2014, Section 4.2)\n");

    // -- Part 1: allocated space --------------------------------------------------
    let plain = CuckooFilter::with_capacity(N, 0.001);
    let semi = SemiSortedCuckooFilter::with_capacity(N, 0.001);

    println!("Part 1: allocated bits per item at the same FPP");
    println!(
        "{:<24} {:>14} {:>16} {:>14}",
        "filter", "fp bits", "bits/bucket", "bits/item"
    );
    let plain_bpi = plain.storage_bytes() as f64 * 8.0 / N as f64;
    let semi_bpi = semi.storage_bytes() as f64 * 8.0 / N as f64;
    println!("{:<24} {:>14} {:>16} {:>14.2}", "plain cuckoo (u16)", 16, 64, plain_bpi);
    println!("{:<24} {:>14} {:>16} {:>14.2}", "semi-sorted (15-bit)", 15, 56, semi_bpi);
    println!("saving: {:.1}% — the paper's one bit per fingerprint.\n", 100.0 * (1.0 - semi_bpi / plain_bpi));

    // -- Part 2: realized FPP ------------------------------------------------------
    let mut filter = SemiSortedCuckooFilter::with_capacity(N, 0.001);
    for i in 0..N {
        filter.insert_item(&i).unwrap();
    }
    let mut false_positives = 0_u64;
    for i in N..N + 100_000 {
        if filter.contains_item(&i) {
            false_positives += 1;
        }
    }
    println!(
        "Part 2: FPP bound 8/2^15 = {:.6}, empirical = {:.6}",
        filter.expected_fpp(),
        false_positives as f64 / 100_000.0
    );

    // -- Part 3: speed for space ------------------------------------------------------
    println!("\nPart 3: operation timing at n = {N}");
    let mut plain = CuckooFilter::with_capacity(N, 0.001);
    let start = Instant::now();
    for i in 0..N {
        plain.insert_item(&i).unwrap();
    }
    let plain_insert = start.elapsed();

    let mut semi = SemiSortedCuckooFilter::with_capacity(N, 0.001);
    let start = Instant::now();
    for i in 0..N {
        semi.insert_item(&i).unwrap();
    }
    let semi_insert = start.elapsed();

    let start = Instant::now();
    let hits: u64 = (0..N).map(|i| plain.contains_item(&i) as u64).sum();
    let plain_query = start.elapsed();
    let start = Instant::now();
    let hits2: u64 = (0..N).map(|i| semi.contains_item(&i) as u64).sum();
    let semi_query = start.elapsed();

    println!("insert: plain {plain_insert:.2?}, semi-sorted {semi_insert:.2?}");
    println!("query:  plain {plain_query:.2?} ({hits}), semi-sorted {semi_query:.2?} ({hits2})");
    println!("\nThe rank/unrank arithmetic makes semi-sorted slower per bucket");
    println!("access; in exchange every bucket is 7 bits smaller.");
}
