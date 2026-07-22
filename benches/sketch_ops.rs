//! Throughput benchmarks for the composed sketches.
//!
//! Insert and query (hit/miss) rates at representative geometries, plus a
//! PackedArray micro-benchmark for the packed-cell read/write path. All
//! input streams are seeded, so runs are comparable across machines and
//! revisions. Run with: `cargo bench`

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkGroup, Criterion, criterion_group, criterion_main};
use criterion::measurement::WallTime;

use adumbratio::block::PackedArray;
use adumbratio::policy::{RngLite, XorShift64};
use adumbratio::sketch::{
    BlockedBloomFilter, BloomFilter, CountMinSketch, CountSketch, CountingBloomFilter,
    CuckooFilter,
};

/// Deterministic pseudo-random items, distinct across seeds.
fn items(n: usize, seed: u64) -> Vec<u64> {
    let mut rng = XorShift64::new(seed);
    (0..n).map(|_| rng.next_u64()).collect()
}

/// Keeps the suite under a couple of minutes while staying statistically
/// meaningful.
fn quick(group: &mut BenchmarkGroup<'_, WallTime>) {
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.sample_size(30);
}

fn bench_bloom(c: &mut Criterion) {
    let mut group = c.benchmark_group("bloom");
    quick(&mut group);

    let present = items(100_000, 1);
    let absent = items(100_000, 2);
    let mut filter = BloomFilter::with_capacity(1_000_000, 0.01);
    for item in &present {
        filter.insert_item(item);
    }

    group.bench_function("insert", |b| {
        let mut i = 0;
        b.iter(|| {
            filter.insert_item(black_box(&present[i % present.len()]));
            i += 1;
        });
    });
    group.bench_function("query_hit", |b| {
        let mut i = 0;
        b.iter(|| {
            let hit = filter.contains_item(black_box(&present[i % present.len()]));
            i += 1;
            black_box(hit)
        });
    });
    group.bench_function("query_miss", |b| {
        let mut i = 0;
        b.iter(|| {
            let hit = filter.contains_item(black_box(&absent[i % absent.len()]));
            i += 1;
            black_box(hit)
        });
    });
    group.finish();
}

fn bench_blocked_bloom(c: &mut Criterion) {
    let mut group = c.benchmark_group("blocked_bloom");
    quick(&mut group);

    let present = items(100_000, 1);
    let absent = items(100_000, 2);
    let mut filter = BlockedBloomFilter::with_capacity(1_000_000, 0.01);
    for item in &present {
        filter.insert_item(item);
    }

    group.bench_function("insert", |b| {
        let mut i = 0;
        b.iter(|| {
            filter.insert_item(black_box(&present[i % present.len()]));
            i += 1;
        });
    });
    group.bench_function("query_hit", |b| {
        let mut i = 0;
        b.iter(|| {
            let hit = filter.contains_item(black_box(&present[i % present.len()]));
            i += 1;
            black_box(hit)
        });
    });
    group.bench_function("query_miss", |b| {
        let mut i = 0;
        b.iter(|| {
            let hit = filter.contains_item(black_box(&absent[i % absent.len()]));
            i += 1;
            black_box(hit)
        });
    });
    group.finish();
}

fn bench_counting_bloom(c: &mut Criterion) {
    let mut group = c.benchmark_group("counting_bloom");
    quick(&mut group);

    let present = items(100_000, 1);
    let mut filter = CountingBloomFilter::with_capacity(1_000_000, 0.01);
    for item in &present {
        filter.insert_item(item);
    }

    group.bench_function("insert", |b| {
        let mut i = 0;
        b.iter(|| {
            filter.insert_item(black_box(&present[i % present.len()]));
            i += 1;
        });
    });
    group.bench_function("query_hit", |b| {
        let mut i = 0;
        b.iter(|| {
            let hit = filter.contains_item(black_box(&present[i % present.len()]));
            i += 1;
            black_box(hit)
        });
    });
    // Remove + reinsert the same item keeps the occupancy steady.
    group.bench_function("remove", |b| {
        let mut i = 0;
        b.iter(|| {
            let item = &present[i % present.len()];
            black_box(filter.remove_item(black_box(item)));
            filter.insert_item(item);
            i += 1;
        });
    });
    group.finish();
}

fn bench_count_min(c: &mut Criterion) {
    let mut group = c.benchmark_group("count_min");
    quick(&mut group);

    let present = items(100_000, 1);
    let mut sketch = CountMinSketch::with_error(0.001, 0.01);
    for item in &present {
        sketch.insert_item(item);
    }

    group.bench_function("insert", |b| {
        let mut i = 0;
        b.iter(|| {
            sketch.insert_item(black_box(&present[i % present.len()]));
            i += 1;
        });
    });
    group.bench_function("estimate", |b| {
        let mut i = 0;
        b.iter(|| {
            let estimate = sketch.estimate_item(black_box(&present[i % present.len()]));
            i += 1;
            black_box(estimate)
        });
    });
    group.finish();
}

fn bench_count_sketch(c: &mut Criterion) {
    let mut group = c.benchmark_group("count_sketch");
    quick(&mut group);

    let present = items(100_000, 1);
    let mut sketch = CountSketch::with_error(0.02, 0.01);
    for item in &present {
        sketch.insert_item(item);
    }

    group.bench_function("insert", |b| {
        let mut i = 0;
        b.iter(|| {
            sketch.insert_item(black_box(&present[i % present.len()]));
            i += 1;
        });
    });
    group.bench_function("estimate", |b| {
        let mut i = 0;
        b.iter(|| {
            let estimate = sketch.estimate_item(black_box(&present[i % present.len()]));
            i += 1;
            black_box(estimate)
        });
    });
    group.finish();
}

fn bench_cuckoo(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuckoo");
    quick(&mut group);

    let present = items(100_000, 1);
    let absent = items(100_000, 2);

    // Size the table so the query load fits in the item vector.
    let mut hot = CuckooFilter::with_capacity(50_000, 0.001);
    let capacity = (hot.geometry().buckets * hot.geometry().slots_per_bucket) as f64;
    let fill = &present[..(0.9 * capacity) as usize];
    for item in fill {
        hot.insert_item(item).expect("90% load inserts must succeed");
    }

    group.bench_function("query_hit_load90", |b| {
        let mut i = 0;
        b.iter(|| {
            let hit = hot.contains_item(black_box(&fill[i % fill.len()]));
            i += 1;
            black_box(hit)
        });
    });
    group.bench_function("query_miss_load90", |b| {
        let mut i = 0;
        b.iter(|| {
            let hit = hot.contains_item(black_box(&absent[i % absent.len()]));
            i += 1;
            black_box(hit)
        });
    });

    // Insert bench at ~50% load, before the kick loop does real work.
    let mut warm = CuckooFilter::with_capacity(50_000, 0.001);
    for item in &present[..(0.5 * capacity) as usize] {
        warm.insert_item(item).expect("50% load inserts must succeed");
    }
    group.bench_function("insert_load50", |b| {
        let mut i = 0;
        b.iter(|| {
            let item = &absent[i % absent.len()];
            let _ = warm.insert_item(black_box(item));
            let _ = warm.remove_item(black_box(item));
            i += 1;
        });
    });
    group.finish();
}

fn bench_hyperloglog(c: &mut Criterion) {
    let mut group = c.benchmark_group("hyperloglog");
    quick(&mut group);

    let present = items(100_000, 1);
    let mut sketch = adumbratio::sketch::HyperLogLog::new(12);
    for item in &present {
        sketch.insert_item(item);
    }

    group.bench_function("insert", |b| {
        let mut i = 0;
        b.iter(|| {
            sketch.insert_item(black_box(&present[i % present.len()]));
            i += 1;
        });
    });
    group.bench_function("cardinality", |b| {
        b.iter(|| black_box(sketch.cardinality()));
    });
    group.finish();
}

fn bench_packed_array(c: &mut Criterion) {
    let mut group = c.benchmark_group("packed_array");
    quick(&mut group);

    let indices: Vec<usize> = {
        let mut rng = XorShift64::new(9);
        (0..100_000).map(|_| rng.next_index(1 << 16)).collect()
    };

    let mut six = PackedArray::<6>::new(1 << 16);
    group.bench_function("get_6bit_straddling", |b| {
        let mut i = 0;
        b.iter(|| {
            let value = six.get(black_box(indices[i % indices.len()]));
            i += 1;
            black_box(value)
        });
    });
    group.bench_function("set_6bit_straddling", |b| {
        let mut i = 0;
        b.iter(|| {
            six.set(black_box(indices[i % indices.len()]), black_box(63));
            i += 1;
        });
    });

    let thirty_two = PackedArray::<32>::new(1 << 16);
    group.bench_function("get_32bit", |b| {
        let mut i = 0;
        b.iter(|| {
            let value = thirty_two.get(black_box(indices[i % indices.len()]));
            i += 1;
            black_box(value)
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_bloom,
    bench_blocked_bloom,
    bench_counting_bloom,
    bench_count_min,
    bench_count_sketch,
    bench_cuckoo,
    bench_hyperloglog,
    bench_packed_array,
);
criterion_main!(benches);
