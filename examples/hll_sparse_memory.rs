//! Shows the HLL++ sparse representation's memory saving at small
//! cardinalities.
//!
//! # What the paper proposes
//!
//! - Stefan Heule, Marc Nunkesser, and Alexander Hall, "HyperLogLog in
//!   Practice: Algorithmic Engineering of a State of the Art Cardinality
//!   Estimation Algorithm", EDBT 2013. <https://doi.org/10.1145/2452376.2452456>
//!   Section 5 ("Sparse Representation"): a fresh HyperLogLog sketch has
//!   almost all registers at zero, so Google's implementation stores only
//!   the non-zero registers as `(index, rho)` pairs while the sketch is
//!   small, converting to the dense array when that stops paying off.
//!   Small sketches — the common case in systems with millions of
//!   independent counters — then cost a few bytes per item instead of the
//!   full register array. Their figure shows memory growing linearly with
//!   cardinality in the sparse regime, then flattening at the dense size.
//!   That is exactly the curve printed below.
//!
//! # What this example does
//!
//! 1. Inserts increasing distinct counts into a `b = 14` sketch and prints
//!    the allocated bytes at each step: 3 bytes per non-zero register in
//!    sparse mode, then a flat 12 KiB after promotion to dense.
//! 2. Confirms the estimates are identical either way — the mode is an
//!    implementation detail, invisible to the API.
//!
//! Inserted items are sequential integers, so the output is reproducible.
//!
//! Run with: `cargo run --release --example hll_sparse_memory`

use adumbratio::sketch::HyperLogLog;

fn main() {
    println!("adumbratio — HLL++ sparse representation (Heule et al. 2013)");
    println!("precision b = 14 -> m = 16384 registers, dense size 12 KiB\n");

    println!("Part 1: memory vs. cardinality");
    println!(
        "{:<12} {:>12} {:>14} {:>12}",
        "distinct n", "mode", "bytes used", "est. error"
    );
    let mut sketch = HyperLogLog::with_seed(14, 5);
    let mut inserted = 0_u64;
    for target in [100_u64, 500, 1_000, 2_000, 3_000, 4_000, 5_000, 10_000, 50_000] {
        while inserted < target {
            sketch.insert_item(&inserted);
            inserted += 1;
        }
        let mode = if sketch.is_sparse() { "sparse" } else { "dense " };
        let error = (sketch.cardinality() - target as f64).abs() / target as f64;
        println!(
            "{:<12} {:>12} {:>14} {:>11.2}%",
            target,
            mode,
            sketch.storage_bytes(),
            error * 100.0
        );
    }
    println!("\nThe HLL++ curve: 3 bytes per non-zero register until m/4 = 4096,");
    println!("then a flat dense array. Small counters cost almost nothing.");

    // -- Part 2: the mode is invisible to the estimate ------------------------------
    let mut whole = HyperLogLog::with_seed(14, 5);
    for i in 0..inserted {
        whole.insert_item(&i);
    }
    println!(
        "\nPart 2: estimate after promotion = {:.1}, rebuilt from all items = {:.1} (identical: {})",
        sketch.cardinality(),
        whole.cardinality(),
        sketch.cardinality() == whole.cardinality()
    );
}
