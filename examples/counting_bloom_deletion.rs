//! Reproduces the counting Bloom filter's deletion workload from the
//! summary-cache paper.
//!
//! # What the paper evaluates
//!
//! - Li Fan, Pei Cao, Jussara Almeida, and Andrei Z. Broder, "Summary
//!   Cache: A Scalable Wide-Area Web Cache Sharing Protocol", IEEE/ACM
//!   Transactions on Networking, 2000. <https://doi.org/10.1109/90.851975>
//!   The counting Bloom filter: small counters instead of bits, so items
//!   can be *deleted* — the feature a web-cache-sharing protocol needs as
//!   documents expire. The paper's two engineering points: deletion
//!   removes items without breaking membership for the rest (no false
//!   negatives as long as you delete only inserted items and counters do
//!   not overflow), and 4-bit counters are enough because counters that
//!   saturate must stay *sticky* — never decremented — so a hot counter
//!   cannot make another item disappear. Both properties are exercised
//!   directly below.
//!
//! # What this example does
//!
//! 1. Fills a filter, deletes half the items, and checks that every
//!    remaining item is still present and the false-positive rate stays at
//!    the bound — the paper's core workload.
//! 2. Demonstrates the sticky-saturation rule: an item inserted many times
//!    pins its counters at 15 (4 bits), and repeated deletions cannot make
//!    it (or anything else) vanish.
//!
//! All inputs are sequential integers, so the output is reproducible.
//!
//! Run with: `cargo run --release --example counting_bloom_deletion`

use adumbratio::sketch::CountingBloomFilter;

const N: u64 = 5_000;

fn main() {
    println!("adumbratio — counting Bloom deletion workloads (Fan et al. 2000)\n");

    // -- Part 1: insert/delete churn ----------------------------------------------
    let mut filter = CountingBloomFilter::with_capacity_and_seed(N, 0.01, 3);
    for i in 0..N {
        filter.insert_item(&i);
    }
    for i in (0..N).step_by(2) {
        filter.remove_item(&i);
    }
    let mut false_negatives = 0_u64;
    for i in (1..N).step_by(2) {
        false_negatives += !filter.contains_item(&i) as u64;
    }
    let mut false_positives = 0_u64;
    for i in N..N + 100_000 {
        false_positives += filter.contains_item(&i) as u64;
    }
    println!("Part 1: {N} inserts, every second one deleted");
    println!("false negatives among survivors: {false_negatives} (guarantee: 0)");
    println!(
        "empirical FPP afterwards: {:.5} (bound: {:.5})",
        false_positives as f64 / 100_000.0,
        filter.expected_fpp(N / 2)
    );

    // -- Part 2: sticky saturation ---------------------------------------------------
    println!("\nPart 2: the sticky-counter rule");
    for _ in 0..40 {
        filter.insert_item(&42_u64);
    }
    println!("inserted '42' 40 times (4-bit counters saturate at 15)");
    let removable: usize = (0..40).map(|_| filter.remove_item(&42_u64) as usize).sum();
    let still_present = filter.contains_item(&42_u64);
    println!(
        "removed it {removable} times; still present: {still_present} (saturated counters never decrement)"
    );
    println!("the trade the paper makes explicit: a hot item may never fully");
    println!("leave, but no other item can disappear because of it.");
}
