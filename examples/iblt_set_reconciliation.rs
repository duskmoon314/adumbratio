//! Reproduces the set-reconciliation scenario of the IBLT papers.
//!
//! # What the papers evaluate
//!
//! - Michael T. Goodrich and Michael Mitzenmacher, "Invertible Bloom
//!   Lookup Tables", Allerton 2011. <https://arxiv.org/abs/1101.2245>
//!   The structural result: with `(count, key_sum, hash_sum)` cells and a
//!   verification hash per key, a Bloom-like table can be *inverted* — its
//!   keys listed by peeling pure cells — when the table holds about
//!   `1.3-1.5x` as many cells as distinct keys.
//!
//! - David Eppstein, Michael T. Goodrich, Frank Uyeda, and George
//!   Varghese, "What's the Difference?: Efficient Set Reconciliation
//!   Without Prior Context", SIGCOMM 2011.
//!   <https://doi.org/10.1145/2018436.2018462>
//!   The systems result that made IBLTs famous: two replicas holding
//!   nearly identical sets can compute their symmetric difference by
//!   subtracting the tables and decoding, exchanging `O(d)` cells for a
//!   difference of size `d` instead of transferring the whole set.
//!
//! # What this example does
//!
//! 1. Builds two "replicas" sharing 9000 of 10_000 keys and differing in
//!    100 keys each, then reconciles them through the IBLT — recovering
//!    the exact divergences with zero errors.
//! 2. Prints the communication cost: the IBLT's cell bytes versus sending
//!    the full 10k-key set — the paper's headline trade, computed with our
//!    own structures.
//! 3. Shows the failure mode honestly: a difference bigger than the table
//!    can hold makes decode fail, which is reported, not silent.
//!
//! All keys are sequential integers, so the output is reproducible.
//!
//! Run with: `cargo run --release --example iblt_set_reconciliation`

use adumbratio::sketch::Iblt;

fn main() {
    println!("adumbratio — IBLT set reconciliation (Goodrich–Mitzenmacher, Eppstein et al.)\n");

    let hasher = adumbratio::hash::DefaultBuildHasher::new(0);

    // -- Part 1: reconcile two replicas -------------------------------------------
    let mut a = Iblt::with_seed(1_500, 0);
    let mut b = Iblt::with_seed(1_500, 0);
    for i in 0..9_000_u64 {
        a.insert_item(&i);
        b.insert_item(&i);
    }
    for i in 9_000..9_100_u64 {
        a.insert_item(&i);
    }
    for i in 9_100..9_200_u64 {
        b.insert_item(&i);
    }

    let reconciliation = a.reconcile(&b).unwrap();
    let only_a_ok = reconciliation
        .only_in_self
        .iter()
        .all(|&h| (9_000..9_100_u64).any(|i| adumbratio::hash::hash_one(&hasher, &i) == h));
    let only_b_ok = reconciliation
        .only_in_other
        .iter()
        .all(|&h| (9_100..9_200_u64).any(|i| adumbratio::hash::hash_one(&hasher, &i) == h));
    println!("Part 1: replicas sharing 9000 keys, 100 divergent each");
    println!(
        "reconciled: {} keys only in A (all correct: {only_a_ok}), {} only in B (all correct: {only_b_ok})",
        reconciliation.only_in_self.len(),
        reconciliation.only_in_other.len()
    );

    // -- Part 2: communication cost --------------------------------------------------
    let cells_bytes = a.storage_bytes();
    let full_set_bytes = 9_200 * 8;
    println!("\nPart 2: bytes exchanged for a 200-key difference");
    println!("IBLT cells (1500 x 16B): {cells_bytes} bytes");
    println!("full set transfer (9200 keys): {full_set_bytes} bytes");
    println!(
        "ratio: {:.1}x — the paper's point: cost follows the difference, not the set.",
        full_set_bytes as f64 / cells_bytes as f64
    );

    // -- Part 3: honest failure mode ---------------------------------------------------
    println!("\nPart 3: when the difference is too big for the table");
    let mut c = Iblt::with_seed(50, 0);
    let d = Iblt::with_seed(50, 0);
    for i in 0..200_u64 {
        c.insert_item(&i);
    }
    // d is empty: the difference is 200 keys in a 75-cell table.
    match c.reconcile(&d) {
        Ok(_) => println!("decoded (unexpected at this load)"),
        Err(_) => println!("decode failed as reported (200 keys in 75 cells cannot peel)"),
    }
    println!("IBLTs are for small differences; bigger ones need bigger tables");
    println!("or a different protocol — the papers say the same.");
}
