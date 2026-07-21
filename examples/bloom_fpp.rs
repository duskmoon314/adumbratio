//! Empirical check of the classical Bloom filter false-positive theory, and
//! of the double-hashing and blocking variants.
//!
//! # What the papers evaluate
//!
//! - Burton H. Bloom, "Space/time trade-offs in hash coding with allowable
//!   errors", CACM 1970. <https://doi.org/10.1145/362686.362692>
//!   The paper's central result is analytical: with `m` bits, `n` inserted
//!   items and `k` hash functions, the false-positive probability is
//!
//!   ```text
//!   p = (1 - (1 - 1/m)^(k*n))^k  ~=  (1 - e^(-k*n/m))^k
//!   ```
//!
//!   minimized at `k = (m/n) * ln 2`. There is nothing to "run" in that
//!   paper; the evaluation tradition since then is to measure the empirical
//!   FPP against this formula.
//!
//! - Adam Kirsch and Michael Mitzenmacher, "Less Hashing, Same Performance:
//!   Building a Better Bloom Filter", ESA 2006.
//!   <https://doi.org/10.1007/11841036_21>
//!   Proves that deriving the `k` positions by double hashing,
//!   `g_i = h1 + i*h2`, loses nothing asymptotically versus `k` independent
//!   hash functions. This crate's default index scheme is exactly double
//!   hashing, so the empirical FPP below should sit right on Bloom's
//!   formula — that agreement is the reproduction.
//!
//! - Felix Putze, Peter Sanders, and Johannes Singler, "Cache-, Hash-, and
//!   Space-Efficient Bloom Filters", ACM JEA.
//!   <https://doi.org/10.1145/1498698.1594230>
//!   Their blocked Bloom filter confines all `k` probes to one cache line.
//!   Their evaluation shows the speedup comes with a slightly *higher* FPP
//!   than the classical formula at the same `m` and `k`, because blocking
//!   distributes bits non-uniformly. The second table checks that
//!   direction: blocked FPP should be close to, but not below, the
//!   classical filter's.
//!
//! # What this example does
//!
//! 1. Fixes `m/n = 10` bits per item and sweeps `k`, printing Bloom's
//!    theoretical FPP next to the measured rate (double hashing, one item
//!    hash). Expect agreement within sampling noise, and the minimum near
//!    `k = 10 * ln 2 ~= 6.9`.
//! 2. Repeats the measurement with the enhanced double-hashing variant on
//!    the optimum geometry.
//! 3. Compares a blocked Bloom filter against the classical one at the same
//!    geometry, expecting a slightly higher but comparable FPP.
//!
//! Methodology follows the standard one: insert `n` distinct items, then
//! query a large number of definitely-absent items and count false
//! positives. Everything is seeded, so the output is reproducible.
//!
//! Run with: `cargo run --release --example bloom_fpp`

use adumbratio::hash::EnhancedDoubleHashing;
use adumbratio::sketch::{BlockedBloomFilter, BloomFilter};

/// Inserted item count. Queries use `4 * N` definitely-absent items, so the
/// empirical FPP standard error at p = 0.01 is about 0.0002 — three decimal
/// places of the printed rates are meaningful.
const N: u64 = 100_000;

fn main() {
    println!("adumbratio — Bloom filter theory vs. practice");
    println!("inserted items n = {N}, bits per item m/n = 10\n");

    // -- Part 1: k sweep -----------------------------------------------------
    // Bloom's formula as a function of k at fixed m/n. The implementation's
    // expected_fpp() reports exactly this formula, so the "theory" column is
    // what the crate itself predicts.
    println!("Part 1: FPP as a function of k (double hashing, one item hash)");
    println!("{:<6} {:>12} {:>12}", "k", "theory", "empirical");
    for k in [2_usize, 4, 5, 6, 7, 8, 10, 12] {
        let m = 10 * N as usize; // m/n = 10 bits per item
        let mut filter = BloomFilter::from_geometry(
            adumbratio::sketch::BloomGeometry { bits: m, hashes: k },
            /* seed */ 42,
        );
        for i in 0..N {
            filter.insert_item(&i);
        }
        let empirical = empirical_fpp(&filter, N);
        println!("{:<6} {:>12.5} {:>12.5}", k, filter.expected_fpp(N), empirical);
    }
    println!("Bloom 1970: minimum near k = (m/n) ln 2 ~= 6.93\n");

    // -- Part 2: index-scheme comparison -------------------------------------
    // Same geometry, three schemes. Kirsch-Mitzenmacher's theorem says plain
    // double hashing should already match the independent-hash formula; the
    // enhanced variant exists for small-m regimes, included here for
    // completeness. BlockedBloomFilter is expected to sit slightly *above*.
    println!("Part 2: same geometry (m/n = 10, k = 7), different index schemes");
    println!("{:<28} {:>12}", "scheme", "empirical FPP");
    let mut plain = BloomFilter::with_capacity_and_seed(N, 0.01, 7);
    fill(&mut plain);
    println!("{:<28} {:>12.5}", "DoubleHashing", empirical_fpp(&plain, N));

    let enhanced = BloomFilter::<_, EnhancedDoubleHashing>::from_parts(
        plain.geometry(),
        7,
        adumbratio::hash::DefaultBuildHasher::new(7),
        EnhancedDoubleHashing,
    );
    let mut enhanced = enhanced;
    fill(&mut enhanced);
    println!(
        "{:<28} {:>12.5}",
        "EnhancedDoubleHashing",
        empirical_fpp(&enhanced, N)
    );

    let mut blocked = BlockedBloomFilter::with_capacity_and_seed(N, 0.01, 7);
    for i in 0..N {
        blocked.insert_item(&i);
    }
    println!(
        "{:<28} {:>12.5}",
        "Blocked (Putze et al.)",
        empirical_fpp_blocked(&blocked, N)
    );
    println!("\nclassical prediction for this geometry: {:.5}", plain.expected_fpp(N));
    println!("Putze et al.: blocking trades a small FPP increase for one cache line per lookup.");
}

fn fill<S, I>(filter: &mut BloomFilter<S, I>)
where
    S: std::hash::BuildHasher,
    I: adumbratio::hash::IndexScheme,
{
    for i in 0..N {
        filter.insert_item(&i);
    }
}

/// Queries `4 * N` items disjoint from the inserted `0..N` and returns the
/// observed false-positive rate.
fn empirical_fpp<S, I>(filter: &BloomFilter<S, I>, n: u64) -> f64
where
    S: std::hash::BuildHasher,
    I: adumbratio::hash::IndexScheme,
{
    let mut false_positives = 0_u64;
    for i in n..5 * n {
        if filter.contains_item(&i) {
            false_positives += 1;
        }
    }
    false_positives as f64 / (4 * n) as f64
}

fn empirical_fpp_blocked<S>(filter: &BlockedBloomFilter<S>, n: u64) -> f64
where
    S: std::hash::BuildHasher,
{
    let mut false_positives = 0_u64;
    for i in n..5 * n {
        if filter.contains_item(&i) {
            false_positives += 1;
        }
    }
    false_positives as f64 / (4 * n) as f64
}
