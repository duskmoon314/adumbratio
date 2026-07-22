//! Reproduces the accuracy law of HyperLogLog cardinality estimation.
//!
//! # What the papers evaluate
//!
//! - Philippe Flajolet, Eric Fusy, Olivier Gandouet, and Frederic Meunier,
//!   "HyperLogLog: the analysis of a near-optimal cardinality estimation
//!   algorithm", AofA 2007. <https://doi.org/10.46298/dmtcs.3545>
//!   The paper is analysis-first, validated by simulation: with `m = 2^b`
//!   registers the relative standard error is `1.04 / sqrt(m)`, essentially
//!   independent of the true cardinality. For small cardinalities the raw
//!   harmonic-mean estimate is biased, and the paper switches to *linear
//!   counting* (estimating from the fraction of empty registers) below
//!   roughly `5m/2`. The first table below measures observed relative
//!   error against the `1.04/sqrt(m)` law across precisions; the second
//!   shows the linear-counting regime staying accurate at small `n`.
//!
//! - Stefan Heule, Marc Nunkesser, and Alexander Hall, "HyperLogLog in
//!   Practice", EDBT 2013. <https://doi.org/10.1145/2452376.2452456>
//!   HLL++ — the engineering follow-up — uses 64-bit hashes so the
//!   large-cardinality bias correction becomes unnecessary in practice.
//!   This crate hashes to 64 bits throughout for the same reason, which is
//!   why the example can run from tiny counts to millions with one code
//!   path.
//!
//! # What this example does
//!
//! 1. For register counts `m` from 256 to 16384, inserts one million
//!    distinct items and prints memory cost, the theoretical standard
//!    error `1.04/sqrt(m)`, and the observed relative error — the paper's
//!    central trade-off, in one table.
//! 2. At a fixed precision, inserts small cardinalities (10..10_000) to
//!    show the linear-counting correction holding accuracy where the raw
//!    harmonic estimate would be biased.
//!
//! Inserted items are sequential integers, so the output is reproducible.
//!
//! Run with: `cargo run --release --example hyperloglog_cardinality`

use adumbratio::sketch::HyperLogLog;

/// Cardinality used for the precision sweep — well into the harmonic-mean
/// regime for every register count below.
const LARGE_N: u64 = 1_000_000;

fn main() {
    println!("adumbratio — HyperLogLog accuracy law (Flajolet et al. 2007)\n");

    // -- Part 1: the 1.04/sqrt(m) law ---------------------------------------
    // The paper's key engineering number: error depends only on m, not on
    // the true cardinality. Each register is a 6-bit packed cell, so memory
    // is 0.75 bytes per register. Following the paper's simulation
    // methodology, each row averages several runs (different hash seeds):
    // a single run scatters around the 1-sigma line, the mean tracks it.
    println!("Part 1: mean relative error vs. 1.04/sqrt(m) at n = {LARGE_N} (5 seeds)");
    println!(
        "{:<12} {:>10} {:>12} {:>14} {:>14}",
        "precision b", "memory", "registers m", "theory 1.04/sqrt(m)", "mean |error|"
    );
    for b in [8_u32, 10, 12, 14] {
        let mut total_error = 0.0_f64;
        let mut sketch = HyperLogLog::with_seed(b, 0);
        for seed in 1..=5_u64 {
            sketch = HyperLogLog::with_seed(b, seed);
            for i in 0..LARGE_N {
                sketch.insert_item(&i);
            }
            total_error += (sketch.cardinality() - LARGE_N as f64).abs() / LARGE_N as f64;
        }
        println!(
            "{:<12} {:>9}B {:>12} {:>14.3}% {:>14.3}%",
            b,
            sketch.storage_bytes(),
            sketch.register_count(),
            sketch.standard_error() * 100.0,
            total_error * 100.0 / 5.0
        );
    }
    println!("Expect the mean at or below the theoretical line; at m = 256 the");
    println!("finite-m error runs slightly above the asymptotic formula.\n");

    // -- Part 2: small cardinalities -----------------------------------------
    // Below ~5m/2 distinct items the raw estimate is biased upward; the
    // paper's linear-counting correction (estimating from empty registers)
    // keeps the error near zero until the registers fill.
    println!("Part 2: small-cardinality accuracy with linear counting (b = 12, m = 4096)");
    println!("{:<12} {:>14} {:>14}", "true n", "estimate", "error");
    for n in [10_u64, 100, 500, 1_000, 5_000, 10_000] {
        let mut sketch = HyperLogLog::with_seed(12, 17);
        for i in 0..n {
            sketch.insert_item(&i);
        }
        let estimate = sketch.cardinality();
        let error = estimate - n as f64;
        println!("{:<12} {:>14.1} {:>+13.1}%", n, estimate, 100.0 * error / n as f64);
    }
    println!("\nThe switch point is ~5m/2 = 10240: up to it, linear counting keeps");
    println!("errors at a fraction of a percent; beyond it the Part-1 law takes over.");
}
