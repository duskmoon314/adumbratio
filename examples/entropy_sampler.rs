//! Demonstrates Shannon entropy estimation with the entropy sampler.
//!
//! # What the technique is
//!
//! - Nick Duffield, Mikkel Thorup, and Carsten Lund, "Priority Sampling
//!   for Estimating Arbitrary Subset Sums", Journal of the ACM, 2007.
//!   <https://doi.org/10.1145/1255443.1255449>
//!   Priority (hashing-based) sampling: a slot that keeps the event with
//!   the largest derived value is a uniform sample over occurrences, and a
//!   uniform sample over occurrences is an item drawn with probability
//!   proportional to its frequency. That makes `E[-log p(sample)] = H`
//!   exactly — Shannon entropy from frequency samples, no distribution
//!   materialization needed.
//!
//! - Zaoxing Liu et al., "One Sketch to Rule Them All: Rethinking Network
//!   Flow Monitoring with UnivMon", SIGCOMM 2016.
//!   <https://doi.org/10.1145/2934872.2934908>
//!   The deployment context: entropy as an anomaly-detection metric in
//!   network monitoring — the workload this example mimics, comparing a
//!   heavy-tailed "attack-like" distribution against a flatter baseline.
//!
//! # What this example does
//!
//! 1. Estimates Shannon entropy for two streams (a Zipf-skewed one and a
//!    uniform one) with an exact oracle, showing the estimate sits on the
//!    truth — each slot is an exact uniform sample, so the only error is
//!    sampling noise, which scales like `1/sqrt(k)`.
//! 2. Replaces the oracle with a Count-Min Sketch, the production
//!    composition: entropy without materializing any distribution, with
//!    the sketch's point error passing through.
//! 3. Shows the estimate tightening as the slot count `k` grows.
//!
//! Streams are seeded, so the output is reproducible.
//!
//! Run with: `cargo run --release --example entropy_sampler`

use std::collections::HashMap;

use adumbratio::hash::{DefaultBuildHasher, hash_one};
use adumbratio::policy::{RngLite, XorShift64};
use adumbratio::sketch::{CountMinSketch, EntropySampler};

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

fn true_entropy(counts: &[u64]) -> f64 {
    let n: f64 = counts.iter().sum::<u64>() as f64;
    -counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / n;
            p * p.log2()
        })
        .sum::<f64>()
}

fn main() {
    println!("adumbratio — Shannon entropy from samples (priority sampling)\n");

    let counts = zipf_counts(10_000, 200_000, 1.1, 11);
    let uniform: Vec<u64> = (0..10_000).map(|_| 20_u64).collect();
    let hasher = DefaultBuildHasher::new(11);

    // -- Part 1: estimate vs. truth for both shapes --------------------------------
    println!("Part 1: estimate vs. truth (k = 1024 slots, exact oracle)");
    println!("{:<14} {:>12} {:>12} {:>12}", "stream", "true H", "estimate", "error");
    for (name, counts) in [("zipf(1.1)", &counts[..]), ("uniform", &uniform[..])] {
        let mut sampler = EntropySampler::with_seed(1_024, 11);
        let mut exact: HashMap<u64, u64> = HashMap::new();
        for (item, &count) in counts.iter().enumerate() {
            for _ in 0..count {
                sampler.insert_item(&(item as u64));
            }
            exact.insert(hash_one(&hasher, &(item as u64)), count);
        }
        let estimate = sampler.shannon_entropy(|hash| *exact.get(&hash).unwrap_or(&0));
        let truth = true_entropy(counts);
        println!(
            "{:<14} {:>12.3} {:>12.3} {:>+11.3}",
            name,
            truth,
            estimate,
            estimate - truth
        );
    }

    // -- Part 2: the production composition with Count-Min ----------------------------
    println!("\nPart 2: same estimate backed by a Count-Min Sketch (no exact counts)");
    let mut sampler = EntropySampler::with_seed(1_024, 11);
    let mut cms = CountMinSketch::with_error_and_seed(0.001, 0.01, 11);
    for (item, &count) in counts.iter().enumerate() {
        for _ in 0..count {
            sampler.insert_item(&(item as u64));
            cms.insert_item(&(item as u64));
        }
    }
    let truth = true_entropy(&counts);
    let estimate = sampler.shannon_entropy(|hash| cms.estimate_hash(hash));
    println!("true H = {truth:.3}, CMS-backed estimate = {estimate:.3}");

    // -- Part 3: noise scaling with k ---------------------------------------------------
    println!("\nPart 3: mean |error| as k grows (5 seeds, exact oracle)");
    println!("{:<10} {:>14}", "k", "mean |error|");
    for k in [64_usize, 256, 1_024, 4_096] {
        let error: f64 = (1..=5_u64)
            .map(|seed| {
                let mut sampler = EntropySampler::with_seed(k, seed);
                let mut exact: HashMap<u64, u64> = HashMap::new();
                let hasher = DefaultBuildHasher::new(seed);
                for (item, &count) in counts.iter().enumerate() {
                    for _ in 0..count {
                        sampler.insert_item(&(item as u64));
                    }
                    exact.insert(hash_one(&hasher, &(item as u64)), count);
                }
                (sampler.shannon_entropy(|hash| *exact.get(&hash).unwrap_or(&0)) - truth).abs()
            })
            .sum::<f64>()
            / 5.0;
        println!("{:<10} {:>14.4}", k, error);
    }
    println!("\nExpected: estimates on the truth, sketch-backed composition within");
    println!("CMS point error, and noise shrinking like 1/sqrt(k).");
}
