//! Reproduces the MinHash estimator behavior for Jaccard similarity.
//!
//! # What the paper evaluates
//!
//! - Andrei Z. Broder, "On the resemblance and containment of documents",
//!   Compression and Complexity of Sequences 1997.
//!   <https://doi.org/10.1109/SEQUEN.1997.666900>
//!   Broder introduced min-wise hashing to measure the resemblance of web
//!   documents (his evaluation ran over an AltaVista crawl). The estimator:
//!   with `k` independent min-hash values per set, the fraction of positions
//!   where two signatures agree is an unbiased estimate of the Jaccard
//!   similarity `J`, with variance `J * (1 - J) / k`. In practice that
//!   means the error shrinks like `1 / sqrt(k)`: four times the hashes
//!   halves the error.
//!
//! # What this example does
//!
//! Instead of web documents we use synthetic sets with controlled overlaps,
//! which makes the *true* Jaccard known exactly — the same idea the paper
//! uses to sanity-check the estimator, but with the ground truth computed
//! rather than measured. Two tables:
//!
//! 1. Estimate vs. true Jaccard at a fixed `k = 256`, with the theoretical
//!    standard error `sqrt(J(1-J)/k)` printed next to the observed
//!    deviation — observed errors should sit within about one sigma.
//! 2. Mean absolute error across signature sizes `k`, showing the
//!    `1 / sqrt(k)` scaling: each doubling of `k` should reduce the error
//!    by about 30%.
//!
//! All sets are sequential integer ranges, so the output is reproducible.
//!
//! Run with: `cargo run --release --example minhash_jaccard`

use adumbratio::sketch::MinHash;

/// Set size for the synthetic pairs; overlaps are computed from the target
/// Jaccard via `c = J * 2s / (1 + J)`.
const SET_SIZE: u64 = 20_000;

/// Builds two sets with a controlled true Jaccard similarity: `c` shared
/// items plus `s - c` private items each, where `c = J * 2s / (1 + J)`.
fn jaccard_pair(k: usize, jaccard: f64, seed: u64) -> (MinHash, MinHash) {
    let c = (jaccard * 2.0 * SET_SIZE as f64 / (1.0 + jaccard)) as u64;
    let mut a = MinHash::with_seed(k, seed);
    let mut b = MinHash::with_seed(k, seed);
    for i in 0..c {
        a.insert_item(&i);
        b.insert_item(&i);
    }
    for i in c..SET_SIZE {
        a.insert_item(&(1_000_000 + i));
        b.insert_item(&(2_000_000 + i));
    }
    (a, b)
}

fn main() {
    println!("adumbratio — MinHash Jaccard estimation (Broder 1997)\n");

    // -- Part 1: estimate vs. truth -------------------------------------------
    // Each row averages five runs with different hash seeds: the estimator
    // is unbiased, so single runs scatter within ~1 sigma while the mean
    // sits on the truth.
    println!("Part 1: estimated vs. true Jaccard (k = 256, 5 seeds)");
    println!(
        "{:<10} {:>12} {:>12} {:>14}",
        "true J", "mean estimate", "mean dev", "theory sigma"
    );
    for jaccard in [0.1_f64, 0.3, 0.5, 0.7, 0.9] {
        let mut estimate_sum = 0.0_f64;
        for seed in 1..=5_u64 {
            let (a, b) = jaccard_pair(256, jaccard, seed);
            estimate_sum += a.jaccard(&b);
        }
        let estimate = estimate_sum / 5.0;
        let sigma = (jaccard * (1.0 - jaccard) / 256.0).sqrt();
        println!(
            "{:<10.1} {:>12.4} {:>+12.4} {:>14.4}",
            jaccard,
            estimate,
            estimate - jaccard,
            sigma
        );
    }

    // -- Part 2: 1/sqrt(k) scaling --------------------------------------------
    // Averaged over the same five similarity levels and three seeds, the
    // mean absolute deviation should shrink by ~1/sqrt(2) per doubling of k.
    println!("\nPart 2: mean absolute deviation as k doubles");
    println!("{:<10} {:>18} {:>18}", "k", "mean |deviation|", "theory max sigma");
    let levels = [0.1_f64, 0.3, 0.5, 0.7, 0.9];
    for k in [64_usize, 128, 256, 512] {
        let mut total = 0.0_f64;
        let mut runs = 0_usize;
        for &j in &levels {
            for seed in 1..=3_u64 {
                let (a, b) = jaccard_pair(k, j, seed);
                total += (a.jaccard(&b) - j).abs();
                runs += 1;
            }
        }
        println!(
            "{:<10} {:>18.4} {:>18.4}",
            k,
            total / runs as f64,
            1.0 / (2.0 * (k as f64).sqrt())
        );
    }
    println!("\nExpected: estimates cluster on the truth within one sigma, and");
    println!("doubling k cuts the deviation by ~30%, the 1/sqrt(k) law.");
}
