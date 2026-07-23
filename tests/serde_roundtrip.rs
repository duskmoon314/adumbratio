//! Serde round-trip tests: a serialized sketch must deserialize to a
//! bit-identical state, preserving answers, occupancy, and merge
//! compatibility (geometry + seed fingerprint travel with the bytes).

#![cfg(feature = "serde")]

use adumbratio::hash::{DefaultBuildHasher, hash_one};
use adumbratio::sketch::{
    AmsSketch, BBitMinHash, BinaryFuseFilter, BlockedBloomFilter, BloomFilter,
    CountMinSketch, CountSketch, CountingBloomFilter, CuckooFilter, DdSketch,
    EntropySampler, HyperLogLog, Iblt, KllSketch, MinHash, MisraGries, QuotientFilter,
    SemiSortedCuckooFilter, SimHash, SpaceSaving, ThetaSketch, TopK, XorFilter,
};
use adumbratio::traits::{EstimateCardinality, Merge};

fn round_trip<T>(sketch: &T) -> T
where
    T: serde::Serialize + for<'de> serde::Deserialize<'de>,
{
    let bytes = postcard::to_allocvec(sketch).expect("serialize");
    postcard::from_bytes(&bytes).expect("deserialize")
}

#[test]
fn bloom_round_trip_preserves_answers_and_merge_compatibility() {
    let mut filter = BloomFilter::with_capacity_and_seed(10_000, 0.01, 42);
    for i in 0..5_000_u64 {
        filter.insert_item(&i);
    }

    let restored: BloomFilter = round_trip(&filter);
    assert_eq!(restored.fill_ratio(), filter.fill_ratio());
    for i in (0..10_000_u64).step_by(3) {
        assert_eq!(restored.contains_item(&i), filter.contains_item(&i));
    }

    // Merge compatibility survives serialization.
    let mut merged = restored;
    merged.merge_from(&filter).expect("merge after round trip");
}

#[test]
fn blocked_bloom_round_trip() {
    let mut filter = BlockedBloomFilter::with_capacity_and_seed(10_000, 0.01, 7);
    for i in 0..5_000_u64 {
        filter.insert_item(&i);
    }
    let restored: BlockedBloomFilter = round_trip(&filter);
    assert_eq!(restored.fill_ratio(), filter.fill_ratio());
    assert_eq!(restored.block_bits(), filter.block_bits());
}

#[test]
fn counting_bloom_round_trip() {
    let mut filter = CountingBloomFilter::with_capacity_and_seed(10_000, 0.01, 3);
    for i in 0..3_000_u64 {
        filter.insert_item(&i);
    }
    let restored: CountingBloomFilter = round_trip(&filter);
    assert_eq!(
        EstimateCardinality::cardinality(&restored),
        EstimateCardinality::cardinality(&filter)
    );
    for i in (0..3_000_u64).step_by(7) {
        assert!(restored.contains_item(&i));
    }
}

#[test]
fn count_min_round_trip_preserves_estimates() {
    let mut sketch = CountMinSketch::with_error_and_seed(0.001, 0.01, 5);
    for i in 0..1_000_u64 {
        for _ in 0..(i % 50) {
            sketch.insert_item(&i);
        }
    }
    let restored: CountMinSketch = round_trip(&sketch);
    assert_eq!(restored.total_count(), sketch.total_count());
    for i in (0..1_000_u64).step_by(11) {
        assert_eq!(restored.estimate_item(&i), sketch.estimate_item(&i));
    }
}

#[test]
fn count_sketch_round_trip_preserves_estimates() {
    let mut sketch = CountSketch::with_error_and_seed(0.02, 0.01, 13);
    for i in 0..1_000_u64 {
        for _ in 0..(i % 30) {
            sketch.insert_item(&i);
        }
    }
    let restored: CountSketch = round_trip(&sketch);
    assert_eq!(restored.total_count(), sketch.total_count());
    for i in (0..1_000_u64).step_by(13) {
        assert_eq!(restored.estimate_signed(&i), sketch.estimate_signed(&i));
    }
}

#[test]
fn hyperloglog_round_trip_preserves_estimate() {
    let mut sketch = HyperLogLog::with_seed(12, 5);
    for i in 0..50_000_u64 {
        sketch.insert_item(&i);
    }
    let restored: HyperLogLog = round_trip(&sketch);
    assert_eq!(restored.cardinality(), sketch.cardinality());
    assert_eq!(restored.precision(), sketch.precision());
}

#[test]
fn minhash_round_trip_preserves_signature() {
    let mut sketch = MinHash::with_seed(256, 3);
    for i in 0..10_000_u64 {
        sketch.insert_item(&i);
    }
    let restored: MinHash = round_trip(&sketch);
    assert_eq!(restored.signature(), sketch.signature());
}

#[test]
fn top_k_round_trip_preserves_candidates() {
    let mut top = TopK::<u64>::new(10, 0.001, 0.01);
    for item in 0..100_u64 {
        for _ in 0..(1000 / (item + 1)) {
            top.insert_item(&item);
        }
    }
    let restored: TopK<u64> = round_trip(&top);
    assert_eq!(restored.top_k(), top.top_k());
    assert_eq!(restored.total_count(), top.total_count());
}

#[test]
fn kll_round_trip_preserves_quantiles() {
    let mut sketch = KllSketch::<u64>::with_seed(100, 3);
    for i in 0..50_000_u64 {
        sketch.insert_item(&(i * 7919 % 50_000));
    }
    let restored: KllSketch<u64> = round_trip(&sketch);
    assert_eq!(restored.count(), sketch.count());
    for q in [0.1, 0.5, 0.9] {
        assert_eq!(restored.quantile(q), sketch.quantile(q));
    }
}

#[test]
fn xor_filter_round_trip_preserves_answers() {
    let items: Vec<u64> = (0..20_000).collect();
    let filter = XorFilter::build_with_seed(&items, 7);
    let restored: XorFilter = round_trip(&filter);
    assert_eq!(restored.len(), filter.len());
    assert_eq!(restored.table_len(), filter.table_len());
    for i in (0..40_000_u64).step_by(3) {
        assert_eq!(restored.contains_item(&i), filter.contains_item(&i));
    }
}

#[test]
fn quotient_filter_round_trip_preserves_answers() {
    let mut filter = QuotientFilter::with_capacity_and_seed(10_000, 0.001, 5);
    for i in 0..7_000_u64 {
        filter.insert_item(&i).unwrap();
    }
    let restored: QuotientFilter = round_trip(&filter);
    assert_eq!(restored.len(), filter.len());
    for i in (0..10_000_u64).step_by(3) {
        assert_eq!(restored.contains_item(&i), filter.contains_item(&i));
    }
}

#[test]
fn theta_round_trip_preserves_estimates() {
    let mut sketch = ThetaSketch::with_seed(256, 3);
    for i in 0..30_000_u64 {
        sketch.insert_item(&i);
    }
    let restored: ThetaSketch = round_trip(&sketch);
    assert_eq!(restored.retained(), sketch.retained());
    assert_eq!(restored.cardinality(), sketch.cardinality());
}

#[test]
fn simhash_round_trip_preserves_signature() {
    let mut sketch = SimHash::with_seed(3);
    for i in 0..10_000_u64 {
        sketch.insert_item(&i);
    }
    let restored: SimHash = round_trip(&sketch);
    assert_eq!(restored.signature(), sketch.signature());
    assert_eq!(restored.sums(), sketch.sums());
}

#[test]
fn ams_round_trip_preserves_counters() {
    let mut sketch = AmsSketch::with_error_and_seed(0.05, 0.01, 5);
    for i in 0..1_000_u64 {
        for _ in 0..(i % 40) {
            sketch.insert_item(&i);
        }
    }
    let restored: AmsSketch = round_trip(&sketch);
    assert_eq!(restored.counters(), sketch.counters());
    assert_eq!(restored.f2(), sketch.f2());
}

#[test]
fn frequent_summaries_round_trip() {
    let mut mg = MisraGries::<u64>::new(10);
    let mut ss = SpaceSaving::<u64>::new(10);
    for item in 0..100_u64 {
        for _ in 0..(100 / (item + 1)) {
            mg.insert_item(&item);
            ss.insert_item(&item);
        }
    }
    let mg_restored: MisraGries<u64> = round_trip(&mg);
    let ss_restored: SpaceSaving<u64> = round_trip(&ss);
    assert_eq!(mg_restored.top_k(), mg.top_k());
    assert_eq!(ss_restored.top_k(), ss.top_k());
    assert_eq!(mg_restored.total_count(), mg.total_count());
    assert_eq!(ss_restored.total_count(), ss.total_count());
}

#[test]
fn binary_fuse_round_trip_preserves_answers() {
    let items: Vec<u64> = (0..20_000).collect();
    let filter = BinaryFuseFilter::build_with_seed(&items, 7);
    let restored: BinaryFuseFilter = round_trip(&filter);
    assert_eq!(restored.len(), filter.len());
    assert_eq!(restored.table_len(), filter.table_len());
    for i in (0..40_000_u64).step_by(3) {
        assert_eq!(restored.contains_item(&i), filter.contains_item(&i));
    }
}

#[test]
fn bbit_minhash_round_trip_preserves_signature() {
    let mut sketch = BBitMinHash::<8>::with_seed(256, 3);
    for i in 0..10_000_u64 {
        sketch.insert_item(&i);
    }
    let restored: BBitMinHash<8> = round_trip(&sketch);
    assert_eq!(restored.signature(), sketch.signature());
}

#[test]
fn ddsketch_round_trip_preserves_quantiles() {
    let mut sketch = DdSketch::new(0.02);
    for i in 1..=10_000_u64 {
        sketch.insert_item(&(i as f64));
    }
    let restored: DdSketch = round_trip(&sketch);
    assert_eq!(restored.count(), sketch.count());
    assert_eq!(restored.buckets(), sketch.buckets());
    assert_eq!(restored.quantile(0.99), sketch.quantile(0.99));
}

#[test]
fn semi_sorted_cuckoo_round_trip_preserves_occupancy() {
    let mut filter = SemiSortedCuckooFilter::with_capacity_and_seed(10_000, 0.001, 9);
    for i in 0..8_000_u64 {
        filter.insert_item(&i).expect("insert below target load");
    }
    let restored: SemiSortedCuckooFilter = round_trip(&filter);
    assert_eq!(restored.occupancy(), filter.occupancy());
    for i in (0..8_000_u64).step_by(5) {
        assert!(restored.contains_item(&i));
    }
}

#[test]
fn iblt_round_trip_preserves_entries() {
    let mut table = Iblt::with_seed(1_000, 3);
    for i in 0..500_u64 {
        table.insert_item(&i);
    }
    let restored: Iblt = round_trip(&table);
    assert_eq!(restored.cell_count(), table.cell_count());
    let mut expected = table.list_entries().unwrap();
    let mut got = restored.list_entries().unwrap();
    expected.sort_unstable();
    got.sort_unstable();
    assert_eq!(got, expected);
}

#[test]
fn top_k_count_sketch_round_trip_preserves_candidates() {
    let mut top = TopK::<u64>::with_count_sketch(10, 0.001, 0.01);
    for item in 0..100_u64 {
        for _ in 0..(1000 / (item + 1)) {
            top.insert_item(&item);
        }
    }
    let restored: TopK<u64, _, adumbratio::sketch::CountSketch> = round_trip(&top);
    assert_eq!(restored.top_k(), top.top_k());
    assert_eq!(restored.total_count(), top.total_count());
}

#[test]
fn entropy_sampler_round_trip_preserves_estimate() {
    let mut sampler = EntropySampler::with_seed(512, 5);
    let mut counts = std::collections::HashMap::new();
    let hasher = DefaultBuildHasher::new(5);
    for i in 0..5_000_u64 {
        sampler.insert_item(&i);
        *counts.entry(hash_one(&hasher, &i)).or_insert(0) += 1;
    }
    let restored: EntropySampler = round_trip(&sampler);
    let oracle = |hash: u64| *counts.get(&hash).unwrap_or(&0);
    assert_eq!(restored.total_count(), sampler.total_count());
    assert_eq!(
        restored.shannon_entropy(oracle),
        sampler.shannon_entropy(oracle)
    );
}

#[test]
fn cuckoo_round_trip_preserves_occupancy() {
    let mut filter = CuckooFilter::with_capacity_and_seed(10_000, 0.001, 9);
    for i in 0..8_000_u64 {
        filter.insert_item(&i).expect("insert below target load");
    }
    let restored: CuckooFilter = round_trip(&filter);
    assert_eq!(restored.occupancy(), filter.occupancy());
    for i in (0..8_000_u64).step_by(5) {
        assert!(restored.contains_item(&i));
    }
}
