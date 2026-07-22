//! Reproduces the heavy-hitter evaluation of the frequent-items literature.
//!
//! # What the papers evaluate
//!
//! - Graham Cormode and Marios Hadjieleftheriou, "Finding Frequent Items in
//!   Data Streams", PVLDB 2008. <https://doi.org/10.14778/1454159.1454225>
//!   The reference evaluation for frequent-item algorithms: synthetic Zipf
//!   streams with known true frequencies, scored by *recall* (how many of
//!   the true top-k are reported) and by the error of the reported counts.
//!   On skewed streams a Count-Min Sketch with a candidate heap — exactly
//!   this crate's [`TopK`] — should report the true heavy hitters with
//!   perfect or near-perfect recall, while the per-item error stays within
//!   the sketch's `eps * N` bound.
//!
//! - Graham Cormode and S. Muthukrishnan, "An Improved Data Stream Summary:
//!   The Count-Min Sketch and its Applications", Journal of Algorithms 2005.
//!   <https://doi.org/10.1016/j.jalgor.2003.12.001>
//!   The underlying point-estimate guarantee that bounds the reported
//!   counts.
//!
//! # What this example does
//!
//! 1. Generates a Zipf(1.1) stream — the skew these papers assume — with
//!    known true frequencies.
//! 2. Prints the true top-10 next to the reported top-10: same items,
//!    counts within `eps * N`.
//! 3. Sweeps `k` and the sketch width, printing recall@k — showing that
//!    recall is perfect for small `k` and degrades only when the frequency
//!    gap near rank `k` drops below the sketch's error bound (an honest
//!    limit of the method, explained in the paper's analysis).
//!
//! The stream is driven by a seeded xorshift, so the output is
//! reproducible. Run with: `cargo run --release --example top_k_heavy_hitters`

use std::collections::HashSet;

use adumbratio::policy::{RngLite, XorShift64};
use adumbratio::sketch::TopK;

/// Stream length `N`: the Count-Min bound is `eps * N`.
const EVENTS: usize = 200_000;

/// Distinct items in the stream.
const UNIVERSE: u64 = 10_000;

fn main() {
    println!("adumbratio — top-k heavy hitters on a Zipf(1.1) stream");
    println!("N = {EVENTS} events over {UNIVERSE} distinct items\n");

    let counts = zipf_counts(UNIVERSE, EVENTS, 1.1, 11);
    let mut order: Vec<usize> = (0..counts.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(counts[i]));

    // -- Part 1: true vs. reported top-10 -------------------------------------
    let mut top = TopK::new(10, 0.001, 0.01);
    for (item, &count) in counts.iter().enumerate() {
        for _ in 0..count {
            top.insert_item(&(item as u64));
        }
    }
    let reported = top.top_k();
    let eps_n = (0.001 * EVENTS as f64) as u64;

    println!("Part 1: top-10, truth vs. report (eps*N = {eps_n})");
    println!(
        "{:<6} {:>10} {:>12} {:>12}",
        "rank", "true item", "true count", "reported est"
    );
    for (rank, &true_item) in order.iter().take(10).enumerate() {
        let (reported_item, estimate) = &reported[rank];
        let marker = if *reported_item == true_item as u64 { "" } else { "  <-- swapped" };
        println!(
            "{:<6} {:>10} {:>12} {:>12}{}",
            rank + 1,
            true_item,
            counts[true_item],
            estimate,
            marker
        );
        debug_assert_eq!(*reported_item, true_item as u64);
    }

    // -- Part 2: recall@k vs. sketch error bound -------------------------------
    // Near rank k the gap between adjacent true counts shrinks; once it is
    // comparable to eps*N the sketch can legitimately swap items across the
    // boundary, so recall@k is the honest metric.
    println!("\nPart 2: recall@k for two sketch sizes");
    println!("{:<8} {:>16} {:>16}", "k", "eps=0.001", "eps=0.01");
    for k in [10_usize, 25, 50, 100] {
        let true_top: HashSet<u64> = order.iter().take(k).map(|&i| i as u64).collect();
        let mut row = [0_usize; 2];
        for (col, eps) in [0.001_f64, 0.01].iter().enumerate() {
            let mut top = TopK::new(k, *eps, 0.01);
            for (item, &count) in counts.iter().enumerate() {
                for _ in 0..count {
                    top.insert_item(&(item as u64));
                }
            }
            row[col] = top
                .top_k()
                .iter()
                .filter(|(item, _)| true_top.contains(item))
                .count();
        }
        println!("{:<8} {:>16} {:>16}", k, row[0], row[1]);
    }
    println!("\nExpected: perfect recall for small k; near rank 100 the true");
    println!("counts are ~tens apart while eps*N is hundreds, so a wider sketch");
    println!("keeps recall high where the narrow one starts swapping boundary items.");
}

/// Simulates a Zipf(s) stream and returns true per-item frequencies, the
/// same construction used in the Count-Min example and in the frequent-items
/// papers' synthetic workloads.
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
