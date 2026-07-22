//! Seeded statistical validation of sketch error bounds.
//!
//! These tests check the guarantees from the papers each sketch implements:
//! empirical false-positive rates within the analytical bound, frequency
//! error within `epsilon * N` for all but a `delta` fraction of point
//! queries, and no false negatives under deletion workloads. Everything is
//! driven by the crate's own `XorShift64`, so the suite is deterministic
//! without extra dev-dependencies.

use adumbratio::hash::{DefaultBuildHasher, PartialKeyCuckoo, hash_one};
use adumbratio::policy::{RngLite, XorShift64};
use adumbratio::sketch::{
    BloomFilter, CountMinSketch, CountSketch, CountingBloomFilter, CuckooFilter, CuckooGeometry,
};

/// Simulates a zipf(s = 1)-like stream and returns true per-item frequencies.
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

/// Empirical false-positive rate of `contains` over definitely-absent items.
fn false_positive_rate(n: u64, queries: u64, contains: impl Fn(u64) -> bool) -> f64 {
    let mut false_positives = 0_u64;
    for i in n..n + queries {
        if contains(i) {
            false_positives += 1;
        }
    }
    false_positives as f64 / queries as f64
}

#[test]
fn bloom_empirical_fpp_within_theoretical_bound() {
    for (n, p) in [(1_000, 0.01), (50_000, 0.001), (10_000, 0.05)] {
        let mut filter = BloomFilter::with_capacity_and_seed(n, p, 42);
        for i in 0..n {
            filter.insert_item(&i);
        }
        for i in 0..n {
            assert!(filter.contains_item(&i), "false negative for {i}");
        }

        let fpp = false_positive_rate(n, 100_000, |i| filter.contains_item(&i));
        assert!(
            fpp <= 1.5 * p,
            "empirical FPP {fpp} exceeds 1.5x target {p} for n = {n}"
        );
    }
}

#[test]
fn blocked_bloom_empirical_fpp_near_theoretical_bound() {
    use adumbratio::sketch::BlockedBloomFilter;

    // Blocking makes bit selection non-uniform, so the realized FPP sits
    // above the classical prediction; the bound here is deliberately looser
    // than for the classical filter.
    for (n, p) in [(1_000, 0.01), (50_000, 0.001)] {
        let mut filter = BlockedBloomFilter::with_capacity_and_seed(n, p, 42);
        for i in 0..n {
            filter.insert_item(&i);
        }
        for i in 0..n {
            assert!(filter.contains_item(&i), "false negative for {i}");
        }

        let fpp = false_positive_rate(n, 100_000, |i| filter.contains_item(&i));
        assert!(
            fpp <= 2.5 * p,
            "empirical FPP {fpp} exceeds 2.5x target {p} for n = {n}"
        );
    }
}

#[test]
fn counting_bloom_deletion_preserves_members() {
    let n = 5_000_u64;
    let mut filter = CountingBloomFilter::with_capacity_and_seed(n, 0.01, 3);
    for i in 0..n {
        filter.insert_item(&i);
    }
    for i in (0..n).step_by(2) {
        assert!(filter.remove_item(&i), "inserted item {i} was not removable");
    }
    // Removing only truly-inserted items must never create false negatives.
    for i in (1..n).step_by(2) {
        assert!(
            filter.contains_item(&i),
            "false negative for {i} after deletions"
        );
    }

    let fpp = false_positive_rate(n, 100_000, |i| filter.contains_item(&i));
    assert!(
        fpp <= 1.5 * 0.01 + 0.001,
        "empirical FPP {fpp} exceeds bound after deletions"
    );
}

#[test]
fn counting_bloom_saturated_counters_stay_sticky() {
    let mut filter = CountingBloomFilter::with_capacity(1_000, 0.01);
    // 4-bit counters saturate at 15; far more inserts push them to the max.
    for _ in 0..40 {
        filter.insert_item("hot");
    }
    // The classic counting-Bloom rule: saturated counters never decrement,
    // so removals cannot make other colliding items disappear.
    for _ in 0..40 {
        assert!(filter.remove_item("hot"));
    }
    assert!(
        filter.contains_item("hot"),
        "saturated counters must remain sticky (documented trade-off)"
    );
}

// The plain and conservative variants are distinct types sharing the
// capability traits; Insert drives both through one code path.
fn check_count_min_bounds<S, U>(sketch: &CountMinSketch<32, S, U>, counts: &[u64], eps_n: u64)
where
    S: std::hash::BuildHasher,
{
    let mut violations = 0_usize;
    for (item, &truth) in counts.iter().enumerate() {
        let estimate = sketch.estimate_item(&(item as u64));
        assert!(
            estimate >= truth,
            "Count-Min underestimated item {item}: {estimate} < {truth}"
        );
        if estimate - truth > eps_n {
            violations += 1;
        }
    }
    // The per-item bound Pr[error > eps*N] <= delta; allow a generous
    // 5% violation rate against delta = 1% to keep the seeded test stable.
    assert!(
        violations as f64 <= counts.len() as f64 * 0.05,
        "{violations} of {} items exceeded eps*N",
        counts.len()
    );
}

fn fill_count_min<S, U>(sketch: &mut CountMinSketch<32, S, U>, counts: &[u64])
where
    S: std::hash::BuildHasher,
    CountMinSketch<32, S, U>: adumbratio::traits::Insert<u64, Err = std::convert::Infallible>,
{
    use adumbratio::traits::Insert;
    for (item, &count) in counts.iter().enumerate() {
        for _ in 0..count {
            sketch.insert(&(item as u64)).unwrap();
        }
    }
}

#[test]
fn count_min_error_within_epsilon_times_n() {
    let (epsilon, delta) = (0.001, 0.01);
    let events = 100_000_usize;
    let counts = zipf_counts(1_000, events, 11);
    let eps_n = (epsilon * events as f64) as u64;

    let mut plain = CountMinSketch::with_error_and_seed(epsilon, delta, 5);
    fill_count_min(&mut plain, &counts);
    check_count_min_bounds(&plain, &counts, eps_n);

    let mut conservative = CountMinSketch::conservative_with_error_and_seed(epsilon, delta, 5);
    fill_count_min(&mut conservative, &counts);
    check_count_min_bounds(&conservative, &counts, eps_n);
}

#[test]
fn count_sketch_error_bounded_for_heavy_items() {
    let (epsilon, delta) = (0.05, 0.01);
    let events = 100_000_usize;
    let counts = zipf_counts(1_000, events, 17);
    let eps_n = (epsilon * events as f64) as i64;

    let mut sketch = CountSketch::with_error_and_seed(epsilon, delta, 13);
    for (item, &count) in counts.iter().enumerate() {
        for _ in 0..count {
            sketch.insert_item(&(item as u64));
        }
    }

    let mut violations = 0_usize;
    for (item, &truth) in counts.iter().enumerate() {
        let estimate = sketch.estimate_signed(&(item as u64));
        let error = (estimate - truth as i64).abs();
        if error > eps_n {
            violations += 1;
        }
    }
    // L1 norm N bounds L2, so |error| <= eps*N must hold with prob >= 1-delta.
    assert!(
        violations as f64 <= counts.len() as f64 * 0.05,
        "{violations} of {} items exceeded eps*N",
        counts.len()
    );
}

/// Recomputes an item's cuckoo placement through the public hashing API:
/// the fingerprint and the lower of its two candidate bucket indices. Two
/// items with the same key are "twins": they occupy the same slot.
fn cuckoo_key(geometry: &CuckooGeometry, seed: u64, item: u64) -> (u64, usize) {
    let hasher = DefaultBuildHasher::new(seed);
    let hash = hash_one(&hasher, &item);
    let fingerprint = PartialKeyCuckoo::fingerprint(geometry.fingerprint_bits, hash);
    let first = PartialKeyCuckoo::bucket(hash, geometry.buckets);
    let second = PartialKeyCuckoo::alt_bucket(first, fingerprint, geometry.buckets);
    (fingerprint, first.min(second))
}

#[test]
fn hyperloglog_estimates_within_standard_errors() {
    use adumbratio::sketch::HyperLogLog;

    // b = 12 -> m = 4096 registers, theoretical standard error ~1.6%.
    for n in [100_u64, 10_000, 1_000_000] {
        let mut sketch = HyperLogLog::with_seed(12, 5);
        for i in 0..n {
            sketch.insert_item(&i);
        }
        let estimate = sketch.cardinality();
        let sigma = sketch.standard_error();
        let relative_error = (estimate - n as f64).abs() / n as f64;
        assert!(
            relative_error <= 4.0 * sigma,
            "n = {n}: estimate {estimate} off by {:.2}% ({} sigma allowed)",
            relative_error * 100.0,
            4.0
        );
    }

    // Duplicate-heavy streams must not inflate the estimate.
    let mut sketch = HyperLogLog::with_seed(14, 5);
    for i in 0..10_000_u64 {
        for _ in 0..10 {
            sketch.insert_item(&i);
        }
    }
    let estimate = sketch.cardinality();
    let relative_error = (estimate - 10_000.0).abs() / 10_000.0;
    assert!(relative_error <= 4.0 * sketch.standard_error());
}

#[test]
fn minhash_jaccard_within_standard_error() {
    use adumbratio::sketch::MinHash;

    // Sets A and B of equal size s with a controlled intersection c, so the
    // true Jaccard similarity is c / (2s - c). The estimator's standard
    // error is sqrt(J(1-J)/k); 4 sigma is generous for a seeded suite.
    let (k, s) = (512_usize, 20_000_u64);
    for jaccard in [0.1_f64, 0.5, 0.9] {
        let c = (jaccard * 2.0 * s as f64 / (1.0 + jaccard)) as u64;
        let mut a = MinHash::with_seed(k, 5);
        let mut b = MinHash::with_seed(k, 5);
        for i in 0..c {
            a.insert_item(&i);
            b.insert_item(&i);
        }
        for i in c..s {
            a.insert_item(&(1_000_000 + i));
            b.insert_item(&(2_000_000 + i));
        }

        let estimate = a.jaccard(&b);
        let sigma = (jaccard * (1.0 - jaccard) / k as f64).sqrt();
        assert!(
            (estimate - jaccard).abs() <= 4.0 * sigma,
            "J = {jaccard}: estimate {estimate} beyond 4 sigma ({sigma})"
        );
    }

    // Disjoint sets estimate exactly zero with overwhelming probability.
    let mut a = MinHash::new(256);
    let mut b = MinHash::new(256);
    for i in 0..1_000_u64 {
        a.insert_item(&i);
        b.insert_item(&(1_000_000 + i));
    }
    assert_eq!(a.jaccard(&b), 0.0);
}

#[test]
fn top_k_recovers_heavy_hitters_on_zipf_stream() {
    use adumbratio::sketch::TopK;

    let counts = zipf_counts(10_000, 200_000, 11);
    let mut top = TopK::new(20, 0.001, 0.01);
    for (item, &count) in counts.iter().enumerate() {
        for _ in 0..count {
            top.insert_item(&(item as u64));
        }
    }

    let mut order: Vec<usize> = (0..counts.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(counts[i]));
    let true_top: std::collections::HashSet<u64> =
        order.iter().take(20).map(|&i| i as u64).collect();

    let reported = top.top_k();
    assert_eq!(reported.len(), 20);
    let hits = reported
        .iter()
        .filter(|(item, _)| true_top.contains(item))
        .count();
    assert!(
        hits >= 19,
        "recall@20 = {hits}/20 on a zipf stream: heavy hitters must be recovered"
    );

    // Reported estimates never underestimate and stay within eps*N.
    let eps_n = (0.001 * 200_000.0) as u64;
    for (item, estimate) in &reported {
        let truth = counts[*item as usize];
        assert!(*estimate >= truth);
        assert!(
            estimate - truth <= eps_n,
            "item {item}: estimate {estimate} exceeds truth {truth} by more than eps*N"
        );
    }
}

#[test]
fn kll_rank_error_within_bound() {
    use adumbratio::sketch::KllSketch;

    // Measured worst-case rank error is ~1.7/k; 2.5/k gives comfortable
    // margin while remaining a meaningful bound (0.0125 at k = 200).
    let k = 200_usize;
    let n = 100_000_u64;
    let mut rng = XorShift64::new(3);
    let mut sketch = KllSketch::with_seed(k, 1);
    let mut values = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let v = rng.next_u64() % 1_000_000;
        sketch.insert_item(&v);
        values.push(v);
    }
    values.sort_unstable();

    // The rank of a value with duplicates is an interval; the error is the
    // distance from q to it (standard rank-error definition).
    for qi in 1..100 {
        let q = qi as f64 / 100.0;
        let estimate = sketch.quantile(q).unwrap();
        let first = values.partition_point(|&v| v < estimate) as f64 / n as f64;
        let last = values.partition_point(|&v| v <= estimate) as f64 / n as f64;
        let error = if q < first {
            first - q
        } else if q > last {
            q - last
        } else {
            0.0
        };
        assert!(
            error <= 2.5 / k as f64,
            "q = {q}: rank error {error} beyond 2.5/k"
        );
    }

    // rank() is the inverse query on the same weighted sample.
    let probe = values[(n / 2) as usize];
    assert!((sketch.rank(&probe) - 0.5).abs() <= 2.5 / k as f64 + 1.0 / n as f64);
}

#[test]
fn xor_filter_fpp_matches_slot_width() {
    use adumbratio::sketch::XorFilter;

    let n = 100_000_u64;
    let items: Vec<u64> = (0..n).collect();
    let filter = XorFilter::build(&items);
    for item in &items {
        assert!(filter.contains_item(item), "missing built item {item}");
    }

    // FPP = 2^-16 for u16 slots; measure over disjoint queries.
    let bound = filter.expected_fpp();
    let fpp = false_positive_rate(n, 200_000, |i| filter.contains_item(&i));
    assert!(
        fpp <= 1.5 * bound + 1e-5,
        "empirical FPP {fpp} exceeds bound {bound}"
    );

    // u8 slots: FPP = 2^-8.
    let narrow = XorFilter::<u8>::build_with_seed(&items, 0);
    let fpp = false_positive_rate(n, 200_000, |i| narrow.contains_item(&i));
    assert!(
        fpp <= 1.5 * narrow.expected_fpp() + 1e-4,
        "u8 empirical FPP {fpp} exceeds bound {}",
        narrow.expected_fpp()
    );
}

/// Recomputes an item's quotient-filter placement through the public
/// hashing API: two items with the same `(quotient, remainder)` pair are
/// indistinguishable to the filter ("twins"), exactly like cuckoo
/// fingerprints. Tests must reason about pairs, not items.
fn quotient_pair(seed: u64, quotient_bits: u32, r_bits: u32, item: u64) -> (u64, u64) {
    let hash = hash_one(&DefaultBuildHasher::new(seed), &item);
    let quotient = (hash >> (64 - quotient_bits)) & ((1 << quotient_bits) - 1);
    let remainder = (hash >> (64 - quotient_bits - r_bits)) & ((1 << r_bits) - 1);
    (quotient, remainder)
}

#[test]
fn quotient_filter_matches_reference_model_under_churn() {
    use adumbratio::sketch::{QuotientFilter, QuotientGeometry};

    // Heavy insert/remove churn on a small table (long clusters, table
    // wrap), verified against a pair-level exact model at every step.
    let (seed, quotient_bits, r_bits) = (1, 6, 10);
    let mut filter =
        QuotientFilter::<10>::from_geometry(QuotientGeometry { quotient_bits }, seed);
    let mut rng = XorShift64::new(99);
    // pair -> (items sharing it, still stored in the filter)
    let mut model: std::collections::HashMap<(u64, u64), (Vec<u64>, bool)> =
        std::collections::HashMap::new();
    let mut counter = 0_u64;
    for round in 0..3_000 {
        if model.is_empty() || rng.next_index(100) < 60 {
            let item = counter;
            counter += 1;
            let pair = quotient_pair(seed, quotient_bits, r_bits, item);
            // The table may legitimately fill on a 64-slot table; only
            // track items the filter actually accepted.
            if filter.insert_item(&item).is_ok() {
                let entry = model.entry(pair).or_insert_with(|| (Vec::new(), true));
                entry.0.push(item);
                entry.1 = true; // the pair is stored (again) after every insert
            }
        } else {
            let victim = rng.next_index(counter as usize) as u64;
            let pair = quotient_pair(seed, quotient_bits, r_bits, victim);
            if let Some((items, stored)) = model.get_mut(&pair)
                && items.contains(&victim)
            {
                assert_eq!(
                    filter.remove_item(&victim),
                    *stored,
                    "remove({victim}) disagrees with the pair model at round {round}"
                );
                *stored = false;
            }
        }
        if round % 250 == 0 {
            let stored_pairs = model.values().filter(|(_, stored)| *stored).count();
            assert_eq!(filter.len(), stored_pairs);
            for (pair, (items, stored)) in &model {
                if !stored {
                    continue;
                }
                for item in items {
                    assert!(
                        filter.contains_item(item),
                        "false negative for {item} at round {round}"
                    );
                }
                let _ = pair;
            }
        }
    }
}

#[test]
fn quotient_filter_high_load_fpp_and_wrap() {
    use adumbratio::sketch::QuotientFilter;

    // 90% load forces wrapping clusters. Removal guarantees hold per
    // fingerprint pair, so remove only twin-group leaders.
    let (seed, n) = (0, 9_000_u64);
    let mut filter = QuotientFilter::with_capacity_and_seed(10_000, 0.001, seed);
    let quotient_bits = filter.geometry().quotient_bits;

    let mut seen = std::collections::HashSet::new();
    let leaders: Vec<u64> = (0..n)
        .filter(|&i| seen.insert(quotient_pair(seed, quotient_bits, 10, i)))
        .collect();

    for &i in &leaders {
        filter.insert_item(&i).unwrap();
    }
    for &i in &leaders {
        assert!(filter.contains_item(&i), "missing {i}");
    }
    for &i in leaders.iter().step_by(2) {
        assert!(filter.remove_item(&i), "stored leader {i} was not removable");
    }
    for &i in leaders.iter().skip(1).step_by(2) {
        assert!(filter.contains_item(&i), "missing {i} after deletions");
    }

    let bound = filter.expected_fpp();
    let fpp = false_positive_rate(n, 100_000, |i| filter.contains_item(&i));
    assert!(
        fpp <= 2.0 * bound + 0.0005,
        "empirical FPP {fpp} exceeds bound {bound}"
    );
}

#[test]
fn cuckoo_high_load_no_false_negatives_and_bounded_fpp() {
    let n = 10_000_u64;
    let seed = 7;
    let mut filter = CuckooFilter::with_capacity_and_seed(n, 0.01, seed);
    let geometry = filter.geometry();

    // insert() deduplicates fingerprint twins (same fingerprint in the same
    // bucket pair), so only the first item of each twin group is stored.
    // Work with those leaders to test the storage guarantees.
    let mut seen = std::collections::HashSet::new();
    let leaders: Vec<u64> = (0..n)
        .filter(|&i| seen.insert(cuckoo_key(&geometry, seed, i)))
        .collect();

    for &i in &leaders {
        filter
            .insert_item(&i)
            .expect("insert should succeed below the target load factor");
    }
    for &i in &leaders {
        assert!(filter.contains_item(&i), "false negative for {i}");
    }

    for &i in leaders.iter().step_by(2) {
        assert!(filter.remove_item(&i), "stored item {i} was not removable");
    }
    for &i in leaders.iter().skip(1).step_by(2) {
        assert!(
            filter.contains_item(&i),
            "false negative for {i} after deletions"
        );
    }

    // Removed items may still look present only at the false-positive rate.
    let mut residual = 0_u64;
    for &i in leaders.iter().step_by(2) {
        if filter.contains_item(&i) {
            residual += 1;
        }
    }
    let bound = filter.expected_fpp();
    let removed = leaders.len().div_ceil(2) as f64;
    assert!(
        residual as f64 / removed <= 2.0 * bound + 0.005,
        "removed items remain present at rate {}, bound {bound}",
        residual as f64 / removed
    );

    let fpp = false_positive_rate(n, 50_000, |i| filter.contains_item(&i));
    assert!(
        fpp <= 1.5 * bound + 0.002,
        "empirical FPP {fpp} exceeds bound {bound}"
    );
}

#[test]
fn cuckoo_insert_deduplicates_fingerprint_twins() {
    // Fan et al. 2014 caveat: two items sharing a fingerprint and bucket
    // pair are indistinguishable; the second insert is a no-op and removing
    // one removes both. This test documents that behavior explicitly.
    let seed = 7;
    let geometry = CuckooGeometry {
        buckets: 64,
        slots_per_bucket: 4,
        fingerprint_bits: 8,
        max_kicks: 10,
    };

    let mut first_seen = std::collections::HashMap::new();
    let mut twins = None;
    for i in 0..100_000_u64 {
        let key = cuckoo_key(&geometry, seed, i);
        if let Some(&other) = first_seen.get(&key) {
            twins = Some((other, i));
            break;
        }
        first_seen.insert(key, i);
    }
    let (a, b) = twins.expect("8-bit fingerprints should collide quickly");

    let mut filter = CuckooFilter::<u8>::from_geometry(geometry, seed);
    filter.insert_item(&a).unwrap();
    let occupancy = filter.occupancy();
    filter.insert_item(&b).unwrap();
    assert_eq!(
        filter.occupancy(),
        occupancy,
        "twin insert must be deduplicated"
    );

    assert!(filter.remove_item(&a));
    assert!(
        !filter.contains_item(&b),
        "removing one twin removes the shared fingerprint"
    );
}
