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

fn bench_minhash(c: &mut Criterion) {
    let mut group = c.benchmark_group("minhash");
    quick(&mut group);

    let present = items(100_000, 1);
    let mut sketch = adumbratio::sketch::MinHash::new(256);
    for item in &present {
        sketch.insert_item(item);
    }

    group.bench_function("insert_k256", |b| {
        let mut i = 0;
        b.iter(|| {
            sketch.insert_item(black_box(&present[i % present.len()]));
            i += 1;
        });
    });
    let other = sketch.clone();
    group.bench_function("jaccard_k256", |b| {
        b.iter(|| black_box(sketch.jaccard(black_box(&other))));
    });
    group.finish();
}

fn bench_top_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("top_k");
    quick(&mut group);

    let present = items(100_000, 1);
    let mut top = adumbratio::sketch::TopK::new(100, 0.001, 0.01);
    for item in &present {
        top.insert_item(item);
    }

    group.bench_function("insert_k100", |b| {
        let mut i = 0;
        b.iter(|| {
            top.insert_item(black_box(&present[i % present.len()]));
            i += 1;
        });
    });
    group.bench_function("top_k_query", |b| {
        b.iter(|| black_box(top.top_k()));
    });
    group.finish();
}

fn bench_iblt(c: &mut Criterion) {
    let mut group = c.benchmark_group("iblt");
    quick(&mut group);

    let present = items(50_000, 1);
    let mut table = adumbratio::sketch::Iblt::with_seed(60_000, 1);
    for item in &present {
        table.insert_item(item);
    }

    group.bench_function("insert", |b| {
        let mut i = 0;
        b.iter(|| {
            table.insert_item(black_box(&present[i % present.len()]));
            i += 1;
        });
    });
    group.bench_function("decode_50k", |b| {
        b.iter(|| black_box(table.list_entries().unwrap()));
    });
    group.finish();
}

fn bench_kll(c: &mut Criterion) {
    let mut group = c.benchmark_group("kll");
    quick(&mut group);

    let present = items(100_000, 1);
    let mut sketch = adumbratio::sketch::KllSketch::with_seed(200, 1);
    for item in &present {
        sketch.insert_item(item);
    }

    group.bench_function("insert_k200", |b| {
        let mut i = 0;
        b.iter(|| {
            sketch.insert_item(black_box(&present[i % present.len()]));
            i += 1;
        });
    });
    group.bench_function("quantile", |b| {
        b.iter(|| black_box(sketch.quantile(black_box(0.95))));
    });
    group.finish();
}

fn bench_xor_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("xor_filter");
    quick(&mut group);

    let present = items(100_000, 1);
    let absent = items(100_000, 2);
    let filter = adumbratio::sketch::XorFilter::build(&present);

    group.bench_function("build_100k", |b| {
        b.iter(|| black_box(adumbratio::sketch::XorFilter::build(black_box(&present))));
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

fn bench_quotient_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("quotient_filter");
    quick(&mut group);

    let present = items(50_000, 1);
    let absent = items(50_000, 2);
    let mut filter = adumbratio::sketch::QuotientFilter::with_capacity(50_000, 0.001);
    for item in &present[..25_000] {
        filter.insert_item(item).unwrap();
    }

    group.bench_function("insert_load50", |b| {
        let mut i = 0;
        b.iter(|| {
            let item = &absent[i % absent.len()];
            let _ = filter.insert_item(black_box(item));
            let _ = filter.remove_item(black_box(item));
            i += 1;
        });
    });
    for item in &present[25_000..] {
        filter.insert_item(item).unwrap();
    }
    group.bench_function("query_hit_load75", |b| {
        let mut i = 0;
        b.iter(|| {
            let hit = filter.contains_item(black_box(&present[i % present.len()]));
            i += 1;
            black_box(hit)
        });
    });
    group.bench_function("query_miss_load75", |b| {
        let mut i = 0;
        b.iter(|| {
            let hit = filter.contains_item(black_box(&absent[i % absent.len()]));
            i += 1;
            black_box(hit)
        });
    });
    group.finish();
}

fn bench_ddsketch(c: &mut Criterion) {
    let mut group = c.benchmark_group("ddsketch");
    quick(&mut group);

    let present: Vec<f64> = items(100_000, 1)
        .into_iter()
        .map(|v| (v % 1_000_000) as f64 + 1.0)
        .collect();
    let mut sketch = adumbratio::sketch::DdSketch::new(0.02);
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
    group.bench_function("quantile", |b| {
        b.iter(|| black_box(sketch.quantile(black_box(0.99))));
    });
    group.finish();
}

fn bench_bbit_minhash(c: &mut Criterion) {
    let mut group = c.benchmark_group("bbit_minhash");
    quick(&mut group);

    let present = items(100_000, 1);
    let mut sketch = adumbratio::sketch::BBitMinHash::new(256);
    for item in &present {
        sketch.insert_item(item);
    }
    let other = sketch.clone();

    group.bench_function("insert_k256", |b| {
        let mut i = 0;
        b.iter(|| {
            sketch.insert_item(black_box(&present[i % present.len()]));
            i += 1;
        });
    });
    group.bench_function("jaccard_k256", |b| {
        b.iter(|| black_box(sketch.jaccard(black_box(&other))));
    });
    group.finish();
}

fn bench_binary_fuse(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_fuse");
    quick(&mut group);

    let present = items(100_000, 1);
    let absent = items(100_000, 2);
    let filter = adumbratio::sketch::BinaryFuseFilter::build(&present);

    group.bench_function("build_100k", |b| {
        b.iter(|| black_box(adumbratio::sketch::BinaryFuseFilter::build(black_box(&present))));
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

fn bench_ams(c: &mut Criterion) {
    let mut group = c.benchmark_group("ams");
    quick(&mut group);

    let present = items(100_000, 1);
    let mut sketch = adumbratio::sketch::AmsSketch::with_error(0.2, 0.01);
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
    group.bench_function("f2", |b| {
        b.iter(|| black_box(sketch.f2()));
    });
    group.finish();
}

fn bench_frequent(c: &mut Criterion) {
    let mut group = c.benchmark_group("frequent");
    quick(&mut group);

    let present = items(100_000, 1);

    let mut mg = adumbratio::sketch::MisraGries::new(100);
    let mut ss = adumbratio::sketch::SpaceSaving::new(100);
    for item in &present {
        mg.insert_item(item);
        ss.insert_item(item);
    }

    group.bench_function("misra_gries_insert_k100", |b| {
        let mut i = 0;
        b.iter(|| {
            mg.insert_item(black_box(&present[i % present.len()]));
            i += 1;
        });
    });
    group.bench_function("space_saving_insert_k100", |b| {
        let mut i = 0;
        b.iter(|| {
            ss.insert_item(black_box(&present[i % present.len()]));
            i += 1;
        });
    });
    group.finish();
}

fn bench_semi_sorted_cuckoo(c: &mut Criterion) {
    let mut group = c.benchmark_group("semi_sorted_cuckoo");
    quick(&mut group);

    let present = items(50_000, 1);
    let absent = items(50_000, 2);
    let mut filter = adumbratio::sketch::SemiSortedCuckooFilter::with_capacity(50_000, 0.001);
    for item in &present {
        filter.insert_item(item).unwrap();
    }

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
    group.bench_function("insert_remove", |b| {
        let mut i = 0;
        b.iter(|| {
            let item = &absent[i % absent.len()];
            let _ = filter.insert_item(black_box(item));
            let _ = filter.remove_item(black_box(item));
            i += 1;
        });
    });
    group.finish();
}

fn bench_simhash(c: &mut Criterion) {
    let mut group = c.benchmark_group("simhash");
    quick(&mut group);

    let present = items(100_000, 1);
    let mut sketch = adumbratio::sketch::SimHash::new();
    let mut other = adumbratio::sketch::SimHash::new();
    for item in &present {
        sketch.insert_item(item);
        other.insert_item(item);
    }

    group.bench_function("insert", |b| {
        let mut i = 0;
        b.iter(|| {
            sketch.insert_item(black_box(&present[i % present.len()]));
            i += 1;
        });
    });
    group.bench_function("hamming_distance", |b| {
        b.iter(|| black_box(sketch.hamming_distance(black_box(&other))));
    });
    group.finish();
}

fn bench_theta(c: &mut Criterion) {
    let mut group = c.benchmark_group("theta");
    quick(&mut group);

    let present = items(100_000, 1);
    let mut sketch = adumbratio::sketch::ThetaSketch::new(512);
    let mut other = adumbratio::sketch::ThetaSketch::new(512);
    for item in &present {
        sketch.insert_item(item);
        other.insert_item(item);
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
    group.bench_function("estimate_intersection", |b| {
        b.iter(|| black_box(sketch.estimate_intersection(black_box(&other))));
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
    bench_ams,
    bench_bbit_minhash,
    bench_binary_fuse,
    bench_bloom,
    bench_blocked_bloom,
    bench_counting_bloom,
    bench_count_min,
    bench_count_sketch,
    bench_cuckoo,
    bench_ddsketch,
    bench_frequent,
    bench_hyperloglog,
    bench_iblt,
    bench_kll,
    bench_minhash,
    bench_quotient_filter,
    bench_semi_sorted_cuckoo,
    bench_simhash,
    bench_theta,
    bench_top_k,
    bench_xor_filter,
    bench_packed_array,
);
criterion_main!(benches);
