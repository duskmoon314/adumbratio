//! Reproduces the core experimental results of the cuckoo filter paper.
//!
//! # What the paper evaluates
//!
//! - Bin Fan, Dave G. Andersen, Michael Kaminsky, and Michael D.
//!   Mitzenmacher, "Cuckoo Filter: Practically Better Than Bloom",
//!   CoNEXT 2014. <https://doi.org/10.1145/2674005.2674994>
//!
//! Section 5 of the paper ("Evaluation") measures three things we reproduce:
//!
//! 1. **False-positive rate vs. occupancy** (their Figure 3): filters are
//!    filled to increasing load factors and queried with items that were
//!    never inserted. The paper's analytical bound is `2b / 2^f` for bucket
//!    size `b` and fingerprint length `f` bits; measured rates sit at or
//!    below it, roughly independent of occupancy until the table is nearly
//!    full.
//! 2. **Maximum achievable load** (their Table 1): with `b = 4` slots per
//!    bucket the filter fills to about 95% before an insertion fails.
//! 3. **Space efficiency vs. Bloom filters** (their Figure 2 / Section
//!    5.2): bits per element required for a target FPP. A standard Bloom
//!    filter needs `-ln(p) / (ln 2)^2` bits per item; a cuckoo filter needs
//!    `f / alpha` where `alpha` is the achievable load. The paper's
//!    headline — cuckoo filters beat Bloom below about 3% FPP — uses their
//!    *semi-sorted* variant (Section 4.2), which packs one fewer bit per
//!    fingerprint; the plain filter's crossover sits nearer 0.5%. The table
//!    below prints both, so the crossover and its cause are visible.
//!
//! # What this example does
//!
//! 1. Fills one filter (b = 4, f = 16 bits via the default `u16`
//!    fingerprints) to occupancies from 50% to 95%, measuring the empirical
//!    FPP at each level against the paper's `2b/2^f` bound.
//! 2. Continues inserting until the kick loop reports the filter full, and
//!    prints the maximum load achieved (expect ~95%, the paper's number).
//! 3. Prints the bits-per-item comparison table for several target FPPs,
//!    reproducing the paper's "better than Bloom below 3%" crossover.
//!
//! Methodology mirrors the paper: query streams consist of items provably
//! disjoint from every inserted item, so every positive answer is a false
//! positive by construction. All inputs are sequential integers, so the
//! output is reproducible.
//!
//! Run with: `cargo run --release --example cuckoo_load_fpp`

use adumbratio::sketch::CuckooFilter;

/// Total slots is 2^17 = 131_072; the progressive fill goes up to 95% of it.
const MAX_ITEMS: u64 = 131_072;

/// Queries per measurement level; standard error at p = 0.001 is ~0.00014.
const QUERIES: u64 = 50_000;

fn main() {
    println!("adumbratio — cuckoo filter paper reproduction (Fan et al. 2014)");
    println!("b = 4 slots/bucket, f = 16-bit fingerprints (u16 slots)\n");

    // The default u16 filter: 8 bytes/bucket, theoretical bound 2*4/2^16.
    let mut filter = CuckooFilter::with_capacity(100_000, 0.001);
    let geometry = filter.geometry();
    let capacity = (geometry.buckets * geometry.slots_per_bucket) as u64;
    let bound = filter.expected_fpp();
    println!(
        "table: {} buckets x {} slots = {} slots; analytical FPP bound 2b/2^f = {:.6}",
        geometry.buckets, geometry.slots_per_bucket, capacity, bound
    );

    // -- Part 1: FPP vs. occupancy ------------------------------------------
    // Insert disjoint items level by level; at each level, measure the FPP
    // with items outside everything inserted so far.
    println!("\nPart 1: empirical FPP at increasing occupancy");
    println!("{:<12} {:>14} {:>14}", "occupancy", "empirical FPP", "2b/2^f bound");
    let mut inserted = 0_u64;
    for target in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95] {
        let level = (target * capacity as f64) as u64;
        while inserted < level {
            filter
                .insert_item(&inserted)
                .expect("insert should succeed below 95% load");
            inserted += 1;
        }
        let fpp = empirical_fpp(&filter, inserted);
        println!(
            "{:<12.2} {:>14.6} {:>14.6}",
            filter.load_factor(),
            fpp,
            bound
        );
    }

    // -- Part 2: maximum achievable load ------------------------------------
    // Keep inserting fresh items until the bounded kick loop gives up. The
    // paper reports ~95% for b = 4; deviations come from our fixed seed.
    let mut failed_at = None;
    for item in inserted..2 * capacity {
        if filter.insert_item(&item).is_err() {
            failed_at = Some(item);
            break;
        }
    }
    println!(
        "\nPart 2: first failed insert after {} items -> max load {:.3} (paper: ~0.95)",
        failed_at.unwrap_or(2 * capacity),
        filter.load_factor()
    );

    // -- Part 3: space per item vs. target FPP -------------------------------
    // Bloom: bits/item = -ln(p) / ln(2)^2 (optimal k).
    // Cuckoo: bits/item = f / 0.95 with the minimal f meeting 2b/2^f <= p.
    // The paper's semi-sorted variant (Section 4.2) stores one fewer bit per
    // fingerprint, moving the crossover against Bloom from ~0.5% to ~3%.
    println!("\nPart 3: bits per item for a target FPP (paper's crossover: < 3% semi-sorted)");
    println!(
        "{:<12} {:>10} {:>14} {:>14} {:>14}",
        "target FPP", "Bloom", "cuckoo min f", "semi-sorted", "cuckoo u16"
    );
    let ln2_sq = std::f64::consts::LN_2.powi(2);
    for p in [0.03_f64, 0.01, 0.001, 0.000_1] {
        let bloom_bits = -p.ln() / ln2_sq;
        let min_f = ((2.0 * 4.0) / p).log2().ceil().max(1.0);
        let cuckoo_min = min_f / 0.95;
        let semi_sorted = (min_f - 1.0).max(1.0) / 0.95;
        let cuckoo_u16 = 16.0 / 0.95;
        println!(
            "{:<12.4} {:>10.2} {:>14.2} {:>14.2} {:>14.2}",
            p, bloom_bits, cuckoo_min, semi_sorted, cuckoo_u16
        );
    }
    println!("\nTakeaway: the semi-sorted variant wins below ~3% FPP (the paper's");
    println!("headline); our plain u16 filter trades some space for deletion support,");
    println!("which Bloom filters do not offer at all.");
}

/// Measures the FPP with items drawn from a range disjoint from every
/// inserted item (inserted items are `0..inserted`, queries start beyond
/// `MAX_ITEMS`), so every positive answer is a false positive.
fn empirical_fpp(filter: &CuckooFilter, inserted: u64) -> f64 {
    debug_assert!(inserted <= MAX_ITEMS);
    let mut false_positives = 0_u64;
    for i in 2 * MAX_ITEMS..2 * MAX_ITEMS + QUERIES {
        if filter.contains_item(&i) {
            false_positives += 1;
        }
    }
    false_positives as f64 / QUERIES as f64
}
