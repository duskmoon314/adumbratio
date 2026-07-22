//! Head-to-head benchmarks against established sketch crates.
//!
//! Same workloads as `sketch_ops`, measured for `adumbratio` and for the
//! `bloomfilter` and `cuckoofilter` crates. Dev-dependencies only; nothing
//! here ships with the library. Run with: `cargo bench --bench comparison`

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkGroup, Criterion, criterion_group, criterion_main};
use criterion::measurement::WallTime;

use adumbratio::policy::{RngLite, XorShift64};
use adumbratio::sketch::{BloomFilter, CuckooFilter};

fn items(n: usize, seed: u64) -> Vec<u64> {
    let mut rng = XorShift64::new(seed);
    (0..n).map(|_| rng.next_u64()).collect()
}

fn quick(group: &mut BenchmarkGroup<'_, WallTime>) {
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.sample_size(30);
}

fn bench_bloom_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("bloom_compare");
    quick(&mut group);

    let present = items(100_000, 1);
    let absent = items(100_000, 2);

    let mut ours = BloomFilter::with_capacity(1_000_000, 0.01);
    let mut theirs = bloomfilter::Bloom::new_for_fp_rate(1_000_000, 0.01).unwrap();
    for item in &present {
        ours.insert_item(item);
        theirs.set(item);
    }

    group.bench_function("adumbratio/insert", |b| {
        let mut i = 0;
        b.iter(|| {
            ours.insert_item(black_box(&present[i % present.len()]));
            i += 1;
        });
    });
    group.bench_function("bloomfilter/insert", |b| {
        let mut i = 0;
        b.iter(|| {
            theirs.set(black_box(&present[i % present.len()]));
            i += 1;
        });
    });
    group.bench_function("adumbratio/query_hit", |b| {
        let mut i = 0;
        b.iter(|| {
            let hit = ours.contains_item(black_box(&present[i % present.len()]));
            i += 1;
            black_box(hit)
        });
    });
    group.bench_function("bloomfilter/query_hit", |b| {
        let mut i = 0;
        b.iter(|| {
            let hit = theirs.check(black_box(&present[i % present.len()]));
            i += 1;
            black_box(hit)
        });
    });
    group.bench_function("adumbratio/query_miss", |b| {
        let mut i = 0;
        b.iter(|| {
            let hit = ours.contains_item(black_box(&absent[i % absent.len()]));
            i += 1;
            black_box(hit)
        });
    });
    group.bench_function("bloomfilter/query_miss", |b| {
        let mut i = 0;
        b.iter(|| {
            let hit = theirs.check(black_box(&absent[i % absent.len()]));
            i += 1;
            black_box(hit)
        });
    });
    group.finish();
}

fn bench_cuckoo_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuckoo_compare");
    quick(&mut group);

    let present = items(50_000, 1);
    let absent = items(50_000, 2);

    let mut ours = CuckooFilter::with_capacity(50_000, 0.001);
    let mut theirs = cuckoofilter::CuckooFilter::<std::collections::hash_map::DefaultHasher>::with_capacity(50_000);
    for item in &present {
        ours.insert_item(item).expect("insert below capacity");
        theirs.add(item).expect("insert below capacity");
    }

    group.bench_function("adumbratio/insert", |b| {
        let mut i = 0;
        b.iter(|| {
            let item = &absent[i % absent.len()];
            let _ = ours.insert_item(black_box(item));
            let _ = ours.remove_item(black_box(item));
            i += 1;
        });
    });
    group.bench_function("cuckoofilter/insert", |b| {
        let mut i = 0;
        b.iter(|| {
            let item = &absent[i % absent.len()];
            let _ = theirs.add(black_box(item));
            let _ = theirs.delete(black_box(item));
            i += 1;
        });
    });
    group.bench_function("adumbratio/query_hit", |b| {
        let mut i = 0;
        b.iter(|| {
            let hit = ours.contains_item(black_box(&present[i % present.len()]));
            i += 1;
            black_box(hit)
        });
    });
    group.bench_function("cuckoofilter/query_hit", |b| {
        let mut i = 0;
        b.iter(|| {
            let hit = theirs.contains(black_box(&present[i % present.len()]));
            i += 1;
            black_box(hit)
        });
    });
    group.finish();
}

criterion_group!(benches, bench_bloom_comparison, bench_cuckoo_comparison);
criterion_main!(benches);
