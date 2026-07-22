//! Serde round-trip tests: a serialized sketch must deserialize to a
//! bit-identical state, preserving answers, occupancy, and merge
//! compatibility (geometry + seed fingerprint travel with the bytes).

#![cfg(feature = "serde")]

use adumbratio::sketch::{
    BlockedBloomFilter, BloomFilter, CountMinSketch, CountSketch, CountingBloomFilter,
    CuckooFilter, HyperLogLog,
};
use adumbratio::traits::{EstimateCardinality, Merge};

fn round_trip<T>(sketch: &T) -> T
where
    T: serde::Serialize + for<'de> serde::Deserialize<'de>,
{
    let bytes = bincode::serialize(sketch).expect("serialize");
    bincode::deserialize(&bytes).expect("deserialize")
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
