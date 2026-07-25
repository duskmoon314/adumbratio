//! Differential churn tests: seeded, interleaved operation sequences that
//! are continuously verified against exact reference models
//! (`HashMap`/`HashSet`/sorted `Vec`).
//!
//! Where `statistical.rs` checks one-shot error bounds on static streams,
//! these tests hammer the structures with thousands of mixed inserts,
//! weighted inserts, removals, merges, and re-inserts, asserting the
//! contract against ground truth at checkpoints every few hundred
//! operations. Small geometries are used on purpose: collisions, table
//! pressure, and mode transitions are where reference-model divergences
//! live. Everything is driven by the crate's own `XorShift64` with fixed
//! seeds, so a failure reproduces exactly with no extra dev-dependencies.
//!
//! `SpaceSaving` assertions use the theory-backed `N/k` error denominator
//! (the recorded error is a past minimum counter, and the minimum of `k`
//! counters never exceeds `N/k`); `error_bound()` returns the same form.

use std::collections::{HashMap, HashSet};

use adumbratio::error::SketchFull;
use adumbratio::hash::{DefaultBuildHasher, PartialKeyCuckoo, hash_one};
use adumbratio::policy::{RngLite, XorShift64};
use adumbratio::sketch::{
    AmsSketch, CountMinSketch, CountSketch, CountingBloomFilter, CuckooFilter, CuckooGeometry,
    DdSketch, HyperLogLog, Iblt, KllSketch, MisraGries, SemiSortedCuckooFilter, SpaceSaving, TopK,
};
use adumbratio::traits::Merge;

/// Simulates a zipf(s = 1)-like stream and returns true per-item frequencies
/// (same generator as `statistical.rs`).
fn zipf_counts(universe: u64, events: usize, seed: u64) -> Vec<u64> {
    let mut cumulative = Vec::with_capacity(universe as usize);
    let mut acc = 0.0_f64;
    for i in 0..universe {
        acc += 1.0 / (i + 1) as f64;
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

/// Cumulative distribution of a zipf(s = 1)-like law over `0..universe`,
/// for drawing weighted streams item by item.
fn zipf_cdf(universe: usize) -> Vec<f64> {
    let mut cumulative = Vec::with_capacity(universe);
    let mut acc = 0.0_f64;
    for i in 0..universe {
        acc += 1.0 / (i + 1) as f64;
        cumulative.push(acc);
    }
    cumulative
}

/// Draws an item from a zipf cdf.
fn sample_zipf(rng: &mut XorShift64, cumulative: &[f64]) -> usize {
    let total = cumulative[cumulative.len() - 1];
    let target = (rng.next_u64() as f64 / u64::MAX as f64) * total;
    cumulative
        .partition_point(|&c| c < target)
        .min(cumulative.len() - 1)
}

/// Rank error of a quantile estimate against the sorted reference: the
/// distance from `q` to the estimate's true rank interval (the standard
/// definition for values with duplicates, as in `statistical.rs`).
fn quantile_rank_error(sorted: &[u64], estimate: u64, q: f64) -> f64 {
    let n = sorted.len() as f64;
    let first = sorted.partition_point(|&v| v < estimate) as f64 / n;
    let last = sorted.partition_point(|&v| v <= estimate) as f64 / n;
    if q < first {
        first - q
    } else if q > last {
        q - last
    } else {
        0.0
    }
}

const QUANTILES: [f64; 6] = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99];

// ---------------------------------------------------------------------------
// 1. Counting Bloom filter churn
// ---------------------------------------------------------------------------

#[test]
fn counting_bloom_churn_matches_hashset_reference() {
    let (capacity, target_fpp, universe) = (3_000_u64, 0.01_f64, 4_000_u64);
    let mut filter = CountingBloomFilter::with_capacity_and_seed(capacity, target_fpp, 1);
    let mut rng = XorShift64::new(41);
    // Exact multiplicity model: removing only confirmed-present keys (one
    // unit at a time) can never create a false negative, because every
    // decrement is matched by that key's own earlier increment.
    let mut model: HashMap<u64, u32> = HashMap::new();
    // Per-key inserts stay well under the 4-bit saturation point (15) in
    // this phase, so removals are exactly reversible.
    const MAX_PER_KEY: u32 = 12;

    for round in 0..4_000_u64 {
        let key = rng.next_index(universe as usize) as u64;
        let count = model.get(&key).copied().unwrap_or(0);
        if count == 0 || rng.next_index(100) < 60 {
            if count < MAX_PER_KEY {
                filter.insert_item(&key);
                model.insert(key, count + 1);
            }
        } else {
            assert!(
                filter.remove_item(&key),
                "round {round}: confirmed member {key} (count {count}) was not removable"
            );
            model.insert(key, count - 1);
        }

        if round % 400 == 399 {
            let mut absent = 0_u64;
            let mut false_positives = 0_u64;
            for key in 0..universe {
                match model.get(&key).copied().unwrap_or(0) {
                    0 => {
                        absent += 1;
                        if filter.contains_item(&key) {
                            false_positives += 1;
                        }
                    }
                    _ => assert!(
                        filter.contains_item(&key),
                        "round {round}: false negative for member {key}"
                    ),
                }
            }
            let fpp = false_positives as f64 / absent.max(1) as f64;
            assert!(
                fpp <= 5.0 * target_fpp,
                "round {round}: absent-key FPP {fpp} exceeds 5x target {target_fpp} \
                 ({false_positives}/{absent})"
            );
        }
    }

    // Saturation phase: pile hundreds of duplicate inserts on a few hot
    // keys so their 4-bit counters pin at 15. Saturated counters are
    // sticky, so every member must survive both the saturation and the
    // subsequent removals.
    for hot in 5_000..5_030_u64 {
        for _ in 0..200 {
            filter.insert_item(&hot);
        }
        model.insert(hot, 200);
    }
    for (&key, &count) in &model {
        if count > 0 {
            assert!(
                filter.contains_item(&key),
                "member {key} lost while counters saturated"
            );
        }
    }
    for hot in 5_000..5_030_u64 {
        for i in 0..200 {
            assert!(
                filter.remove_item(&hot),
                "saturated hot key {hot} not removable on removal {i}"
            );
        }
    }
    for (&key, &count) in &model {
        if count > 0 {
            assert!(
                filter.contains_item(&key),
                "member {key} lost after hot-key removals"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Cuckoo filter churn at high load (plain and semi-sorted)
// ---------------------------------------------------------------------------

/// The shared surface the churn driver needs from a cuckoo-family filter.
trait ChurnFilter {
    fn insert(&mut self, item: &u64) -> Result<(), SketchFull>;
    fn contains(&self, item: &u64) -> bool;
    fn remove(&mut self, item: &u64) -> bool;
    fn expected_fpp(&self) -> f64;
    fn load_factor(&self) -> f64;
}

impl ChurnFilter for CuckooFilter<u16> {
    fn insert(&mut self, item: &u64) -> Result<(), SketchFull> {
        self.insert_item(item)
    }
    fn contains(&self, item: &u64) -> bool {
        self.contains_item(item)
    }
    fn remove(&mut self, item: &u64) -> bool {
        self.remove_item(item)
    }
    fn expected_fpp(&self) -> f64 {
        self.expected_fpp()
    }
    fn load_factor(&self) -> f64 {
        self.load_factor()
    }
}

impl ChurnFilter for SemiSortedCuckooFilter {
    fn insert(&mut self, item: &u64) -> Result<(), SketchFull> {
        self.insert_item(item)
    }
    fn contains(&self, item: &u64) -> bool {
        self.contains_item(item)
    }
    fn remove(&mut self, item: &u64) -> bool {
        self.remove_item(item)
    }
    fn expected_fpp(&self) -> f64 {
        self.expected_fpp()
    }
    fn load_factor(&self) -> f64 {
        self.load_factor()
    }
}

/// Recomputes an item's cuckoo identity through the public hashing API:
/// the fingerprint and its normalized bucket pair. Two items are "twins"
/// (indistinguishable to the filter) iff these triples agree.
fn cuckoo_pair(seed: u64, fingerprint_bits: u32, buckets: usize, item: u64) -> (u64, usize, usize) {
    let hash = hash_one(&DefaultBuildHasher::new(seed), &item);
    let fp = PartialKeyCuckoo::fingerprint(fingerprint_bits, hash);
    let first = PartialKeyCuckoo::bucket(hash, buckets);
    let second = PartialKeyCuckoo::alt_bucket(first, fp, buckets);
    (fp, first.min(second), first.max(second))
}

/// Handles one rejected insert against the model: the bounded kick loop
/// drops the one fingerprint left in hand; every other stored group must
/// survive. Returns the number of model groups lost (never more than one).
fn note_insert_failure<F: ChurnFilter>(
    label: &str,
    filter: &F,
    model: &mut HashMap<(u64, usize, usize), (Vec<u64>, bool)>,
    round: u64,
    full: SketchFull,
) -> u64 {
    let lost: Vec<(u64, usize, usize)> = model
        .iter()
        .filter(|(_, (items, stored))| *stored && !filter.contains(&items[0]))
        .map(|(&pair, _)| pair)
        .collect();
    assert!(
        lost.len() <= 1,
        "{label} round {round}: one failed insert lost {} member groups",
        lost.len()
    );
    for pair in &lost {
        assert_eq!(
            pair.0,
            full.orphaned_fingerprint(),
            "{label} round {round}: lost group's fingerprint {:#x} is not the \
             reported orphan {:#x}",
            pair.0,
            full.orphaned_fingerprint()
        );
        model.get_mut(pair).expect("lost pair is modeled").1 = false;
    }
    // The rejected item's own fingerprint may or may not have survived the
    // kick chain; the model conservatively leaves its group untracked (an
    // under-approximation, never asserted absent).
    lost.len() as u64
}

/// Heavy insert/remove churn on a small table at high load, verified
/// against a twin-aware pair-level reference model:
///
/// - zero false negatives for every stored pair group, at checkpoints;
/// - a failed insert (`SketchFull`) never panics and loses *at most one*
///   previously stored group — the one carrying the orphaned fingerprint
///   reported in the error (the standard bounded-kick eviction trade-off,
///   surfaced through `SketchFull::orphaned_fingerprint`);
/// - removals only target stored groups and must succeed; a removal may
///   legitimately clear a *different* group sharing the victim's
///   fingerprint value in a shared bucket (the pair-level caveat the
///   filter documents), which the model tracks explicitly;
/// - the absent-key FPP stays bounded throughout the churn.
fn cuckoo_pair_churn<F: ChurnFilter>(
    label: &str,
    filter: &mut F,
    seed: u64,
    fingerprint_bits: u32,
    buckets: usize,
) {
    let mut rng = XorShift64::new(seed ^ 0x5eed);
    // pair -> (items sharing it, still stored in the filter)
    let mut model: HashMap<(u64, usize, usize), (Vec<u64>, bool)> = HashMap::new();
    let mut counter = 0_u64;
    let mut failures = 0_u64;
    let mut clobbers = 0_u64;
    let mut max_load = 0.0_f64;

    // Phase 1: fill the table to the first insert failure, so the churn
    // phase operates at 90%+ load from the start instead of drifting up.
    loop {
        let item = counter;
        counter += 1;
        let pair = cuckoo_pair(seed, fingerprint_bits, buckets, item);
        match filter.insert(&item) {
            Ok(()) => {
                let entry = model.entry(pair).or_insert_with(|| (Vec::new(), true));
                entry.0.push(item);
                entry.1 = true;
            }
            Err(full) => {
                failures += 1;
                note_insert_failure(label, filter, &mut model, counter, full);
                break;
            }
        }
        assert!(counter < 20_000, "{label}: table never filled");
        max_load = max_load.max(filter.load_factor());
    }

    // Phase 2: churn slightly insert-heavy so the table keeps pressing
    // against capacity and failures keep occurring.
    for round in 0..3_000_u64 {
        if rng.next_index(100) < 55 {
            let item = counter;
            counter += 1;
            let pair = cuckoo_pair(seed, fingerprint_bits, buckets, item);
            match filter.insert(&item) {
                Ok(()) => {
                    let entry = model.entry(pair).or_insert_with(|| (Vec::new(), true));
                    entry.0.push(item);
                    entry.1 = true; // the pair is stored (again) after every accepted insert
                }
                Err(full) => {
                    failures += 1;
                    note_insert_failure(label, filter, &mut model, round, full);
                }
            }
        } else {
            let stored: Vec<(u64, usize, usize)> = model
                .iter()
                .filter(|(_, entry)| entry.1)
                .map(|(&pair, _)| pair)
                .collect();
            let pair = stored[rng.next_index(stored.len())];
            let victim = model[&pair].0[0];
            assert!(
                filter.remove(&victim),
                "{label} round {round}: stored group member {victim} was not removable"
            );
            model.get_mut(&pair).expect("victim pair is modeled").1 = false;
            // Removing the victim clears *a* matching fingerprint in its
            // buckets; a twin-value group whose slot sat there goes with it.
            let fp = pair.0;
            let affected: Vec<(u64, usize, usize)> = model
                .iter()
                .filter(|(other, entry)| {
                    **other != pair && other.0 == fp && entry.1 && !filter.contains(&entry.0[0])
                })
                .map(|(&other, _)| other)
                .collect();
            for other in affected {
                model.get_mut(&other).expect("affected pair is modeled").1 = false;
                clobbers += 1;
            }
        }
        max_load = max_load.max(filter.load_factor());

        if round % 250 == 249 {
            for (&pair, (items, stored)) in &model {
                if !stored {
                    continue;
                }
                for item in items {
                    assert!(
                        filter.contains(item),
                        "{label} round {round}: false negative for {item} (pair {pair:?})"
                    );
                }
            }
            let queries = 2_000_u64;
            let mut false_positives = 0_u64;
            for i in 1_000_000..1_000_000 + queries {
                if filter.contains(&i) {
                    false_positives += 1;
                }
            }
            let bound = filter.expected_fpp();
            let fpp = false_positives as f64 / queries as f64;
            assert!(
                fpp <= 5.0 * bound + 0.001,
                "{label} round {round}: absent-key FPP {fpp} exceeds bound {bound} \
                 ({false_positives}/{queries})"
            );
        }
    }

    assert!(
        failures > 0,
        "{label}: the churn never filled the table; raise the op count"
    );
    assert!(
        max_load >= 0.90,
        "{label}: churn only reached {:.1}% load, expected 90%+",
        max_load * 100.0
    );
    // Fingerprint-collision removals are legal at the pair level, so
    // `clobbers` is only tracked, not bounded; it must stay rare.
    assert!(
        clobbers <= failures / 10 + 20,
        "{label}: {clobbers} same-fingerprint clobbers in {failures} failures is implausible"
    );
}

#[test]
fn cuckoo_churn_matches_pair_model_at_high_load() {
    let (seed, buckets) = (7_u64, 128_usize);
    let geometry = CuckooGeometry {
        buckets,
        slots_per_bucket: 4,
        fingerprint_bits: 16,
        max_kicks: 500,
    };
    let mut filter = CuckooFilter::<u16>::from_geometry(geometry, seed);
    cuckoo_pair_churn("cuckoo<u16>", &mut filter, seed, 16, buckets);
}

#[test]
fn semi_sorted_cuckoo_churn_matches_pair_model_at_high_load() {
    let (seed, buckets) = (11_u64, 128_usize);
    // Semi-sorted cuckoo filters fix the fingerprint width at 15 bits.
    let geometry = CuckooGeometry {
        buckets,
        slots_per_bucket: 4,
        fingerprint_bits: 15,
        max_kicks: 500,
    };
    let mut filter = SemiSortedCuckooFilter::from_geometry(geometry, seed);
    cuckoo_pair_churn("semi-sorted", &mut filter, seed, 15, buckets);
}

// ---------------------------------------------------------------------------
// 3. Count-Min / Count Sketch weighted growth churn
// ---------------------------------------------------------------------------

#[test]
fn count_min_and_count_sketch_match_hashmap_under_weighted_growth() {
    let universe = 2_000_usize;
    let events = 24_000_usize;
    let cdf = zipf_cdf(universe);
    let mut rng = XorShift64::new(23);

    let mut plain = CountMinSketch::with_error_and_seed(0.001, 0.01, 5);
    let mut conservative = CountMinSketch::conservative_with_error_and_seed(0.001, 0.01, 5);
    // Count Sketch additionally runs as two half-stream sketches merged at
    // the end: the merge is counter-wise addition, so the merged estimates
    // must equal the sequential ones *exactly*.
    let mut sequential = CountSketch::with_error_and_seed(0.005, 0.01, 13);
    let mut left = CountSketch::with_error_and_seed(0.005, 0.01, 13);
    let mut right = CountSketch::with_error_and_seed(0.005, 0.01, 13);

    let mut truth = vec![0_u64; universe];
    let mut weighted_n = 0_u64;
    for event in 0..events {
        let item = sample_zipf(&mut rng, &cdf);
        let weight = 1 + rng.next_index(1_000) as u64;
        truth[item] += weight;
        weighted_n += weight;
        plain.insert_count(&item, weight);
        conservative.insert_count(&item, weight);
        sequential.insert_count(&item, weight);
        if event < events / 2 {
            left.insert_count(&item, weight);
        } else {
            right.insert_count(&item, weight);
        }

        if event % 500 == 499 {
            let slack = 5.0 * 0.005 * weighted_n as f64;
            for (item, &exact) in truth.iter().enumerate() {
                let est_plain = plain.estimate_item(&item);
                let est_conservative = conservative.estimate_item(&item);
                assert!(
                    est_plain >= exact,
                    "event {event}: plain underestimated {item}: {est_plain} < {exact}"
                );
                assert!(
                    est_conservative >= exact,
                    "event {event}: conservative update underestimated {item}: \
                     {est_conservative} < {exact}"
                );
                // Conservative update increments a subset of the counters
                // plain update touches, so it can never estimate higher.
                assert!(
                    est_conservative <= est_plain,
                    "event {event}: conservative {est_conservative} > plain {est_plain} for {item}"
                );
                let signed = sequential.estimate_signed(&item);
                let error = (signed - exact as i64).abs();
                assert!(
                    error as f64 <= slack,
                    "event {event}: Count Sketch error {error} for {item} exceeds 5*eps*N \
                     ({slack:.0}; signed estimate {signed}, exact {exact})"
                );
            }
        }
    }
    assert_eq!(plain.total_count(), weighted_n);
    assert_eq!(sequential.total_count(), weighted_n);

    // Differential merge: counter addition is linear, so merged halves are
    // the sequential sketch.
    left.merge_from(&right).expect("same geometry and seed");
    assert_eq!(left.total_count(), weighted_n);
    for (item, &exact) in truth.iter().enumerate() {
        let merged = left.estimate_signed(&item);
        let one_shot = sequential.estimate_signed(&item);
        assert_eq!(
            merged, one_shot,
            "merged Count Sketch disagrees with the sequential sketch for {item}: \
             {merged} vs {one_shot} (exact {exact})"
        );
    }
}

// ---------------------------------------------------------------------------
// 4. Misra-Gries / Space-Saving weighted churn
// ---------------------------------------------------------------------------

#[test]
fn misra_gries_and_space_saving_honor_weighted_deterministic_bounds() {
    let k = 25_usize;
    let universe = 500_usize;
    let events = 4_000_usize;
    let cdf = zipf_cdf(universe);
    let mut rng = XorShift64::new(31);

    let mut mg_weighted = MisraGries::new(k);
    // The same item stream as unit events: the documented N/(k+1) bounds
    // are proven for unit inserts and are verified here on `mg_unit`.
    let mut mg_unit = MisraGries::new(k);
    let mut ss = SpaceSaving::new(k);

    let mut weighted_truth = vec![0_u64; universe];
    let mut unit_truth = vec![0_u64; universe];
    let mut weighted_n = 0_u64;
    let mut unit_n = 0_u64;

    for event in 0..events {
        let item = sample_zipf(&mut rng, &cdf) as u64;
        let weight = 1 + rng.next_index(1_000) as u64;
        weighted_truth[item as usize] += weight;
        unit_truth[item as usize] += 1;
        weighted_n += weight;
        unit_n += 1;
        mg_weighted.insert_count(&item, weight);
        mg_unit.insert_item(&item);
        ss.insert_count(&item, weight);

        if event % 250 == 249 {
            let unit_bound = unit_n / (k as u64 + 1);
            let weighted_bound = weighted_n / (k as u64 + 1);
            // Space-Saving's recorded error is a past value of the minimum
            // counter; the minimum of k counters never exceeds N/k.
            let ss_bound = weighted_n / k as u64;
            for item in 0..universe as u64 {
                let index = item as usize;

                // Weighted Misra-Gries is exactly unit-iterated-equivalent,
                // so the full [f - N/(k+1), f] bound and the presence
                // guarantee hold with weighted N.
                let est_weighted = mg_weighted.estimate_item(&item);
                let exact_weighted = weighted_truth[index];
                assert!(
                    est_weighted <= exact_weighted
                        && est_weighted + weighted_bound >= exact_weighted,
                    "event {event}: weighted MG estimate {est_weighted} for {item} outside \
                     [{exact_weighted} - {weighted_bound}, {exact_weighted}]"
                );
                if exact_weighted > weighted_bound {
                    assert!(
                        est_weighted > 0,
                        "event {event}: weighted MG lost {item} with count {exact_weighted} > {weighted_bound}"
                    );
                }

                let est_unit = mg_unit.estimate_item(&item);
                let exact_unit = unit_truth[index];
                assert!(
                    est_unit <= exact_unit && est_unit + unit_bound >= exact_unit,
                    "event {event}: unit MG estimate {est_unit} for {item} outside \
                     [{exact_unit} - {unit_bound}, {exact_unit}]"
                );
                if exact_unit > unit_bound {
                    assert!(
                        est_unit > 0,
                        "event {event}: unit MG lost {item} with count {exact_unit} > {unit_bound}"
                    );
                }

                let (count, error) = ss.estimate_with_error(&item);
                let exact = weighted_truth[index];
                if count > 0 {
                    assert!(
                        count >= exact && count <= exact + error,
                        "event {event}: SS count {count} error {error} vs exact {exact} for {item}"
                    );
                }
                assert!(
                    error <= ss_bound,
                    "event {event}: SS error {error} for {item} exceeds N/k ({ss_bound})"
                );
                if exact > ss_bound {
                    assert!(
                        count > 0,
                        "event {event}: SS lost {item} with count {exact} > N/k ({ss_bound})"
                    );
                }
            }
        }
    }
    assert_eq!(mg_weighted.total_count(), weighted_n);
    assert_eq!(mg_unit.total_count(), unit_n);
    assert_eq!(ss.total_count(), weighted_n);
}

// ---------------------------------------------------------------------------
// 5. Top-k weighted churn (Count-Min and Count Sketch backends)
// ---------------------------------------------------------------------------

/// Elephants and mice: 20 elephants, each weighing thousands of times a
/// unit-weight mouse, interleaved with thousands of mice. At checkpoints
/// the reference `HashMap` decides the true top-k; the reported list must
/// contain all of it.
fn top_k_elephant_churn(label: &str, top: &mut impl TopKChurn) {
    let (elephants, mice, events) = (20_usize, 2_000_usize, 12_000_usize);
    let mut rng = XorShift64::new(47);
    let mut truth = vec![0_u64; elephants + mice];

    for event in 0..events {
        // 25% of events are elephant traffic carrying heavy weights (some
        // as bulk `insert_count` bursts, all well above any mouse total).
        if rng.next_index(100) < 25 {
            let e = rng.next_index(elephants);
            let weight = 1_000 * (e as u64 + 1) + rng.next_index(1_000) as u64;
            top.insert_count(&e, weight);
            truth[e] += weight;
        } else {
            let m = elephants + rng.next_index(mice);
            top.insert_item(&m);
            truth[m] += 1;
        }

        if event % 1_000 == 999 {
            let mut order: Vec<usize> = (0..truth.len()).collect();
            order.sort_by_key(|&i| std::cmp::Reverse(truth[i]));
            let kth_true = truth[order[19]];
            let true_top: HashSet<usize> = order.iter().take(20).copied().collect();

            let reported = top.top_k();
            assert_eq!(
                reported.len(),
                20,
                "{label} event {event}: expected 20 candidates, got {}",
                reported.len()
            );
            let hits = reported
                .iter()
                .filter(|(item, _)| true_top.contains(item))
                .count();
            assert_eq!(
                hits, 20,
                "{label} event {event}: recall {hits}/20 against the reference top-k"
            );
            for (item, _) in &reported {
                assert!(
                    truth[*item] * 4 >= kth_true,
                    "{label} event {event}: reported {item} with count {} far below the true \
                     20th count {kth_true}",
                    truth[*item]
                );
            }
        }
    }
}

/// The churn surface shared by both TopK backends (kept out of the generic
/// parameters so one driver covers Count-Min and Count Sketch).
trait TopKChurn {
    fn insert_item(&mut self, item: &usize);
    fn insert_count(&mut self, item: &usize, count: u64);
    fn top_k(&self) -> Vec<(usize, u64)>;
}

impl TopKChurn for TopK<usize> {
    fn insert_item(&mut self, item: &usize) {
        TopK::insert_item(self, item);
    }
    fn insert_count(&mut self, item: &usize, count: u64) {
        TopK::insert_count(self, item, count);
    }
    fn top_k(&self) -> Vec<(usize, u64)> {
        TopK::top_k(self)
    }
}

impl TopKChurn for TopK<usize, DefaultBuildHasher, CountSketch> {
    fn insert_item(&mut self, item: &usize) {
        TopK::insert_item(self, item);
    }
    fn insert_count(&mut self, item: &usize, count: u64) {
        TopK::insert_count(self, item, count);
    }
    fn top_k(&self) -> Vec<(usize, u64)> {
        TopK::top_k(self)
    }
}

#[test]
fn top_k_recovers_elephants_under_weighted_churn() {
    let mut top = TopK::with_seed(20, 0.001, 0.01, 5);
    top_k_elephant_churn("top-k<count-min>", &mut top);

    let mut top_cs = TopK::<usize>::with_count_sketch_and_seed(20, 0.001, 0.01, 5);
    top_k_elephant_churn("top-k<count-sketch>", &mut top_cs);
}

// ---------------------------------------------------------------------------
// 6. KLL / DDSketch / AMS: merged halves vs one-shot on the same stream
// ---------------------------------------------------------------------------

#[test]
fn kll_merged_halves_match_one_shot_rank_bounds() {
    let k = 100_usize;
    let n = 60_000_usize;
    let mut rng = XorShift64::new(3);
    let mut values = Vec::with_capacity(n);
    let mut one_shot = KllSketch::with_seed(k, 1);
    let mut left = KllSketch::with_seed(k, 1);
    let mut right = KllSketch::with_seed(k, 1);
    for i in 0..n {
        let v = rng.next_u64() % 1_000_000;
        values.push(v);
        one_shot.insert_item(&v);
        if i < n / 2 {
            left.insert_item(&v);
        } else {
            right.insert_item(&v);
        }
    }
    left.merge_from(&right).expect("equal k");
    assert_eq!(left.count(), n as u64);
    values.sort_unstable();

    // Merging adds its own compactions, so merged and one-shot are not
    // identical; both must independently honor the rank bound. The static
    // KLL test measures worst-case error ~1.7/k; 4/k is generous headroom.
    let bound = 4.0 / k as f64;
    for q in QUANTILES {
        for (label, sketch) in [("merged", &left), ("one-shot", &one_shot)] {
            let estimate = sketch.quantile(q).expect("non-empty sketch");
            let error = quantile_rank_error(&values, estimate, q);
            assert!(
                error <= bound,
                "KLL {label} q = {q}: rank error {error} beyond 4/k ({bound})"
            );
        }
    }
}

#[test]
fn ddsketch_merged_halves_are_structurally_identical_to_one_shot() {
    let alpha = 0.02_f64;
    let n = 60_000_usize;
    let mut rng = XorShift64::new(7);
    let mut values = Vec::with_capacity(n);
    let mut one_shot = DdSketch::new(alpha);
    let mut left = DdSketch::new(alpha);
    let mut right = DdSketch::new(alpha);
    for i in 0..n {
        let log_value = rng.next_u64() as f64 / u64::MAX as f64 * 6.0;
        let value = 10_f64.powf(log_value);
        values.push(value);
        one_shot.insert_item(&value);
        if i < n / 2 {
            left.insert_item(&value);
        } else {
            right.insert_item(&value);
        }
    }
    left.merge_from(&right).expect("equal gamma");

    // DDSketch bucketing is deterministic per value, so the merged buckets
    // are exactly the one-shot buckets — no probabilistic slack at all.
    assert_eq!(
        left.buckets(),
        one_shot.buckets(),
        "merged DDSketch buckets differ from the one-shot buckets"
    );
    values.sort_by(f64::total_cmp);
    for q in QUANTILES {
        let estimate = one_shot.quantile(q).expect("non-empty sketch");
        let truth = values[(q * (n - 1) as f64) as usize];
        let ratio = estimate / truth;
        assert!(
            (1.0 - 4.0 * alpha..=1.0 + 4.0 * alpha).contains(&ratio),
            "DDSketch q = {q}: estimate {estimate} vs truth {truth} (ratio {ratio}) \
             outside 1 +/- 4*alpha"
        );
    }
}

#[test]
fn ams_merged_halves_are_counter_identical_to_one_shot() {
    // Same zipf stream and error slack as the static AMS test.
    let counts = zipf_counts(1_000, 100_000, 11);
    let truth: f64 = counts.iter().map(|&c| (c as f64) * (c as f64)).sum();

    let mut one_shot = AmsSketch::with_error_and_seed(0.2, 0.01, 5);
    let mut left = AmsSketch::with_error_and_seed(0.2, 0.01, 5);
    let mut right = AmsSketch::with_error_and_seed(0.2, 0.01, 5);
    let mut event = 0_usize;
    for (item, &count) in counts.iter().enumerate() {
        for _ in 0..count {
            one_shot.insert_item(&(item as u64));
            if event < 50_000 {
                left.insert_item(&(item as u64));
            } else {
                right.insert_item(&(item as u64));
            }
            event += 1;
        }
    }
    left.merge_from(&right).expect("same geometry and seed");

    // AMS counters are linear in the stream, so merged halves are the
    // sequential sketch exactly.
    assert_eq!(left.total_count(), 100_000);
    assert_eq!(
        left.counters(),
        one_shot.counters(),
        "merged AMS counters differ from the one-shot counters"
    );

    // Both halves of the comparison must still meet the static test's F2
    // slack (they are identical here, so this guards the stream setup).
    for (label, sketch) in [("merged", &left), ("one-shot", &one_shot)] {
        let estimate = sketch.f2();
        let relative = (estimate - truth).abs() / truth;
        assert!(
            relative <= 0.15,
            "AMS {label}: F2 estimate {estimate} vs truth {truth} off by {:.2}%",
            relative * 100.0
        );
    }
}

// ---------------------------------------------------------------------------
// 7. HyperLogLog sparse -> dense transition churn
// ---------------------------------------------------------------------------

#[test]
fn hyperloglog_sparse_dense_transition_churn() {
    // b = 14: m = 16384 registers; the sparse representation promotes to
    // dense once more than m/4 = 4096 registers are set, so a 12k-distinct
    // stream straddles the transition. sigma = 1.04/sqrt(m).
    let mut sketch = HyperLogLog::with_seed(14, 5);
    let mut rng = XorShift64::new(61);
    let n = 12_000_u64;
    let mut saw_sparse = false;
    let mut saw_dense = false;

    for i in 0..n {
        sketch.insert_item(&i);
        // Re-inserting an already-seen item touches the same register with
        // the same rho: the state — and therefore the estimate — must not
        // move at all.
        if i % 7 == 3 {
            let seen = rng.next_index(i as usize) as u64;
            let before = sketch.cardinality();
            sketch.insert_item(&seen);
            let after = sketch.cardinality();
            assert_eq!(
                before, after,
                "re-insert of {seen} at n = {} moved the estimate: {before} -> {after}",
                i + 1
            );
        }

        if i % 600 == 599 {
            let exact = (i + 1) as f64;
            let estimate = sketch.cardinality();
            let sigma = sketch.standard_error();
            let relative = (estimate - exact).abs() / exact;
            assert!(
                relative <= 4.0 * sigma,
                "n = {}: estimate {estimate} off by {:.2}% (4 sigma = {:.2}%)",
                i + 1,
                relative * 100.0,
                4.0 * sigma * 100.0
            );
            saw_sparse |= sketch.is_sparse();
            saw_dense |= !sketch.is_sparse();
        }
    }

    assert!(saw_sparse, "the churn never entered sparse mode");
    assert!(saw_dense, "the churn never left sparse mode");
}

// ---------------------------------------------------------------------------
// 8. IBLT reconciliation churn and capacity honesty
// ---------------------------------------------------------------------------

#[test]
fn iblt_churn_reconciles_exactly_within_capacity() {
    let seed = 3_u64;
    let hasher = DefaultBuildHasher::new(seed);
    // Decode load is the symmetric difference (capped at 160 keys below);
    // 300 cells give 1.9x headroom over the worst-case difference, near
    // the paper's 1.3-1.5x rule of thumb for k = 4 distinct cells per key.
    let mut a = Iblt::with_seed(200, seed);
    let mut b = Iblt::with_seed(200, seed);
    let mut ref_a: HashSet<u64> = HashSet::new();
    let mut ref_b: HashSet<u64> = HashSet::new();
    for i in 0..800_u64 {
        a.insert_item(&i);
        b.insert_item(&i);
        ref_a.insert(i);
        ref_b.insert(i);
    }
    for i in 800..840_u64 {
        a.insert_item(&i);
        ref_a.insert(i);
    }
    for i in 840..880_u64 {
        b.insert_item(&i);
        ref_b.insert(i);
    }

    let mut rng = XorShift64::new(67);
    let mut moves = 0_u64;
    for round in 0..60 {
        for _ in 0..3 {
            let item = rng.next_index(800) as u64;
            let diff = ref_a.symmetric_difference(&ref_b).count();
            // Churn one core item on a random side, bounded so the
            // difference stays well within decode capacity.
            if rng.next_index(2) == 0 {
                if ref_a.contains(&item) && diff < 160 {
                    a.remove_item(&item);
                    ref_a.remove(&item);
                    moves += 1;
                } else if !ref_a.contains(&item) {
                    a.insert_item(&item);
                    ref_a.insert(item);
                    moves += 1;
                }
            } else if ref_b.contains(&item) && diff < 160 {
                b.remove_item(&item);
                ref_b.remove(&item);
                moves += 1;
            } else if !ref_b.contains(&item) {
                b.insert_item(&item);
                ref_b.insert(item);
                moves += 1;
            }
        }

        if round % 6 == 5 {
            let reconciliation = a.reconcile(&b).expect("diff within capacity decodes");
            let only_a: HashSet<u64> = reconciliation.only_in_self.into_iter().collect();
            let only_b: HashSet<u64> = reconciliation.only_in_other.into_iter().collect();
            let want_a: HashSet<u64> = ref_a
                .difference(&ref_b)
                .map(|i| hash_one(&hasher, i))
                .collect();
            let want_b: HashSet<u64> = ref_b
                .difference(&ref_a)
                .map(|i| hash_one(&hasher, i))
                .collect();
            assert_eq!(
                only_a, want_a,
                "round {round}: only_in_self disagrees with the reference difference"
            );
            assert_eq!(
                only_b, want_b,
                "round {round}: only_in_other disagrees with the reference difference"
            );
            for &member in &ref_a {
                assert!(
                    a.contains_item(&member),
                    "round {round}: false negative for member {member}"
                );
            }
        }
    }
    assert!(moves > 0, "the churn performed no set mutations");
}

#[test]
fn iblt_reconcile_never_reports_a_wrong_difference() {
    let seed = 9_u64;
    let hasher = DefaultBuildHasher::new(seed);
    // A ladder of difference sizes from comfortable to hopeless. Capacity
    // honesty: decode is all-or-nothing behind a verification hash and an
    // all-cells-empty check, so a returned `Ok` must always be *exactly*
    // the reference symmetric difference, and a difference far beyond the
    // cell count must surface a reported failure — never a partial or
    // wrong set presented as success. With distinct cell positions per
    // key, every difference with at least ~3x headroom must decode.
    let mut decodes = 0_u64;
    for diff in [20_u64, 60, 120, 200, 400, 1_200] {
        let mut a = Iblt::with_seed(400, seed);
        let mut b = Iblt::with_seed(400, seed);
        for i in 0..1_000_u64 {
            a.insert_item(&i);
            b.insert_item(&i);
        }
        for i in 0..diff / 2 {
            a.insert_item(&(2_000 + i));
            b.insert_item(&(3_000 + i));
        }

        let want_a: HashSet<u64> = (2_000..2_000 + diff / 2)
            .map(|i| hash_one(&hasher, &i))
            .collect();
        let want_b: HashSet<u64> = (3_000..3_000 + diff / 2)
            .map(|i| hash_one(&hasher, &i))
            .collect();
        let result = a.reconcile(&b);
        if diff <= 200 {
            assert!(
                result.is_ok(),
                "diff {diff}: 600 cells give >=3x headroom; decode must succeed"
            );
        }
        if diff == 1_200 {
            // 8 mappings per cell cannot peel; anything this far beyond
            // capacity MUST fail loudly rather than misreport.
            assert!(
                result.is_err(),
                "diff {diff}: 1_200 keys in a 600-cell table must not decode"
            );
        }
        if let Ok(reconciliation) = result {
            decodes += 1;
            let only_a: HashSet<u64> = reconciliation.only_in_self.into_iter().collect();
            let only_b: HashSet<u64> = reconciliation.only_in_other.into_iter().collect();
            assert_eq!(
                only_a, want_a,
                "diff {diff}: reconcile reported a WRONG only_in_self set"
            );
            assert_eq!(
                only_b, want_b,
                "diff {diff}: reconcile reported a WRONG only_in_other set"
            );
        }
    }
    assert!(
        decodes >= 4,
        "only {decodes}/6 ladder points decoded; anti-vacuousness floor"
    );
}
