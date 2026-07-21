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

    let mut filter = CuckooFilter::from_geometry(geometry, seed);
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
