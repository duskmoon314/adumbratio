//! Reproduces the b-bit MinHash storage/accuracy trade-off.
//!
//! # What the paper evaluates
//!
//! - Ping Li and Arnd Christian König, "b-Bit Minwise Hashing", WWW 2010.
//!   <https://doi.org/10.1145/1772690.1772759>
//!   The paper's central experiment: comparing Jaccard estimates from
//!   full-width minwise signatures against signatures truncated to the
//!   lowest `b` bits of each minimum, on document pairs of known
//!   similarity. Truncation adds a collision term — two different minima
//!   agree in `b` bits with probability `2^-b` — which the estimator
//!   corrects: `J = (r - 2^-b) / (1 - 2^-b)`. The result that made the
//!   technique standard: one byte per minimum loses almost nothing at the
//!   near-duplicate similarities people actually query (J >= 0.5), while
//!   storage drops 8x.
//!
//! # What this example does
//!
//! 1. Prints the estimator's mean error against the truth at b = 4, 8, 16
//!    and at full 64-bit width, for three similarity levels — the paper's
//!    table shape, with our own construction.
//! 2. Prints the signature size at each width: the trade-off in one row.
//!
//! All sets are sequential integer ranges, so the output is reproducible.
//!
//! Run with: `cargo run --release --example bbit_minhash_storage`

use adumbratio::sketch::{BBitMinHash, MinHash};

const SET_SIZE: u64 = 20_000;
const K: usize = 512;

/// Builds two sets with a controlled true Jaccard similarity at width `B`.
/// Non-default widths construct via `from_parts`, the generic entry point.
fn jaccard_pair_bbit<const B: u32>(
    k: usize,
    jaccard: f64,
    seed: u64,
) -> (BBitMinHash<B>, BBitMinHash<B>) {
    let build = || {
        let hasher = adumbratio::hash::DefaultBuildHasher::new(seed);
        BBitMinHash::<B>::from_parts(k, hasher.seed_fingerprint(), hasher)
    };
    let (mut a, mut b) = (build(), build());
    let c = (jaccard * 2.0 * SET_SIZE as f64 / (1.0 + jaccard)) as u64;
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

fn jaccard_pair_full(k: usize, jaccard: f64, seed: u64) -> (MinHash, MinHash) {
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
    println!("adumbratio — b-bit MinHash trade-off (Li–König 2010)");
    println!("k = {K} minima per signature, mean error over 3 seeds\n");

    println!("Part 1: mean |estimate - truth| at several widths and similarities");
    println!(
        "{:<10} {:>14} {:>14} {:>14} {:>14}",
        "J", "b = 4", "b = 8", "b = 16", "full (64)"
    );
    for jaccard in [0.1_f64, 0.5, 0.9] {
        let e4: f64 = (1..=3_u64)
            .map(|seed| {
                let (a, b) = jaccard_pair_bbit::<4>(K, jaccard, seed);
                (a.jaccard(&b) - jaccard).abs()
            })
            .sum::<f64>()
            / 3.0;
        let e8: f64 = (1..=3_u64)
            .map(|seed| {
                let (a, b) = jaccard_pair_bbit::<8>(K, jaccard, seed);
                (a.jaccard(&b) - jaccard).abs()
            })
            .sum::<f64>()
            / 3.0;
        let e16: f64 = (1..=3_u64)
            .map(|seed| {
                let (a, b) = jaccard_pair_bbit::<16>(K, jaccard, seed);
                (a.jaccard(&b) - jaccard).abs()
            })
            .sum::<f64>()
            / 3.0;
        let e64: f64 = (1..=3_u64)
            .map(|seed| {
                let (a, b) = jaccard_pair_full(K, jaccard, seed);
                (a.jaccard(&b) - jaccard).abs()
            })
            .sum::<f64>()
            / 3.0;
        println!(
            "{:<10.1} {:>14.4} {:>14.4} {:>14.4} {:>14.4}",
            jaccard, e4, e8, e16, e64
        );
    }

    println!("\nPart 2: signature size per width");
    println!("{:<10} {:>16}", "width", "signature bytes");
    for (name, bytes) in [
        ("b = 4", K * 4 / 8),
        ("b = 8", K * 8 / 8),
        ("b = 16", K * 16 / 8),
        ("full (64)", K * 64 / 8),
    ] {
        println!("{:<10} {:>16}", name, bytes);
    }
    println!("\nThe paper's conclusion in our numbers: b = 8 tracks the full");
    println!("signature closely at J >= 0.5 while the signature drops 8x.");
}
