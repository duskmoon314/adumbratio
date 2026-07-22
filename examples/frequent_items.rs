//! Compares the two deterministic heavy-hitter summaries on a Zipf stream.
//!
//! # What the papers establish
//!
//! - Jayadev Misra and David Gries, "Finding repeated elements", Science
//!   of Computer Programming 2(2), 1982.
//!   <https://doi.org/10.1016/0167-6423(82)90012-0>
//!   The decrement-all summary: k counters, every item above `N/(k+1)` is
//!   tracked, and estimates never *over*estimate.
//!
//! - Richard M. Karp, Scott Shenker, and Christos H. Papadimitriou, "A
//!   Simple Algorithm for Finding Frequent Elements in Streams and Bags",
//!   ACM TODS 2003. <https://doi.org/10.1145/762471.762473>
//!   The rediscovery and clean analysis used for the guarantees printed
//!   below: `f(x) - N/(k+1) <= estimate(x) <= f(x)`.
//!
//! - Ahmed Metwally, Divyakant Agrawal, and Amr El Abbadi, "Efficient
//!   Computation of Frequent and Top-k Elements in Data Streams", ICDT
//!   2005. <https://doi.org/10.1007/978-3-540-30570-5_27>
//!   The dual construction: replace-the-minimum keeps counts that never
//!   *under*estimate, with a recorded per-item error bounded by `N/(k+1)`.
//!   On skewed traffic Space-Saving is the more accurate of the two in
//!   practice — visible in the table below.
//!
//! # What this example does
//!
//! Runs a Zipf(1.1) stream (200k events) through both summaries at
//! `k = 20` and prints the top-10 reported by each next to the truth, plus
//! the guarantee checks the papers prove: every item above the bound is
//! tracked, and every estimate lies within its deterministic interval.
//!
//! The stream is seeded, so the output is reproducible.
//!
//! Run with: `cargo run --release --example frequent_items`

use adumbratio::policy::{RngLite, XorShift64};
use adumbratio::sketch::{MisraGries, SpaceSaving};

const EVENTS: usize = 200_000;
const K: usize = 20;

fn main() {
    println!("adumbratio — deterministic heavy hitters (Misra–Gries vs. Space-Saving)");
    println!("N = {EVENTS} events, Zipf(1.1), k = {K} counters\n");

    let counts = zipf_counts(10_000, EVENTS, 1.1, 11);
    let mut mg = MisraGries::new(K);
    let mut ss = SpaceSaving::new(K);
    for (item, &count) in counts.iter().enumerate() {
        for _ in 0..count {
            mg.insert_item(&(item as u64));
            ss.insert_item(&(item as u64));
        }
    }
    let bound = EVENTS as u64 / (K as u64 + 1);
    println!("deterministic error bound N/(k+1) = {bound}\n");

    // -- Part 1: top-10 side by side ----------------------------------------------
    let mut order: Vec<usize> = (0..counts.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(counts[i]));
    let mg_top = mg.top_k();
    let ss_top = ss.top_k();

    println!("Part 1: top-10, truth vs. both summaries");
    println!(
        "{:<6} {:>10} {:>12} {:>12} {:>12}",
        "rank", "true item", "true count", "MG estimate", "SS estimate"
    );
    for (rank, &item) in order.iter().take(10).enumerate() {
        let mg_est = mg_top
            .iter()
            .find(|(candidate, _)| *candidate == item as u64)
            .map(|(_, e)| *e)
            .unwrap_or(0);
        let ss_est = ss_top
            .iter()
            .find(|(candidate, _)| *candidate == item as u64)
            .map(|(_, e)| *e)
            .unwrap_or(0);
        println!(
            "{:<6} {:>10} {:>12} {:>12} {:>12}",
            rank + 1,
            item,
            counts[item],
            mg_est,
            ss_est
        );
    }

    // -- Part 2: the guarantees, checked over the whole universe -------------------
    let mut mg_ok = true;
    let mut ss_ok = true;
    let mut above_tracked_mg = 0_usize;
    let mut above_tracked_ss = 0_usize;
    let mut above = 0_usize;
    for (item, &truth) in counts.iter().enumerate() {
        let item = item as u64;
        if truth > bound {
            above += 1;
            above_tracked_mg += (mg.estimate_item(&item) > 0) as usize;
            above_tracked_ss += (ss.estimate_item(&item) > 0) as usize;
        }
        let mg_est = mg.estimate_item(&item);
        mg_ok &= mg_est <= truth && mg_est + bound >= truth;
        let (count, error) = ss.estimate_with_error(&item);
        if count > 0 {
            ss_ok &= count >= truth && count <= truth + error && error <= bound;
        }
    }
    println!("\nPart 2: guarantee checks over all {} items", counts.len());
    println!("MG: all estimates in [f-bound, f] = {mg_ok}; items above bound tracked: {above_tracked_mg}/{above}");
    println!("SS: all estimates in [f, f+err], err <= bound = {ss_ok}; items above bound tracked: {above_tracked_ss}/{above}");
    println!("\nReading the table: MG reports lower bounds (never over), SS reports");
    println!("upper bounds (never under) and is typically tighter on skewed streams.");
}

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
