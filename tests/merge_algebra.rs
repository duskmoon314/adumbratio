//! Merge-algebra validation: `merge(a, b)` must behave like one sketch that
//! saw both streams, and merging must commute. Same seed and geometry make
//! sketches bit-deterministic, so merged and directly-built sketches answer
//! identically.

use adumbratio::error::MergeError;
use adumbratio::sketch::{
    BloomFilter, BloomGeometry, CountMinGeometry, CountMinSketch, CountSketch,
    CountSketchGeometry, CountingBloomFilter,
};
use adumbratio::traits::Merge;

/// Deterministic per-item frequencies with moderate skew.
fn stream_counts(universe: u64) -> Vec<u64> {
    (0..universe).map(|i| (i * 7919) % 500 + 1).collect()
}

#[test]
fn bloom_merge_matches_single_filter_and_commutes() {
    let geometry = BloomGeometry {
        bits: 1 << 16,
        hashes: 7,
    };
    let mut left = BloomFilter::from_geometry(geometry, 9);
    let mut right = BloomFilter::from_geometry(geometry, 9);
    let mut single = BloomFilter::from_geometry(geometry, 9);

    for i in 0..10_000_u64 {
        left.insert_item(&i);
        single.insert_item(&i);
    }
    for i in 10_000..20_000_u64 {
        right.insert_item(&i);
        single.insert_item(&i);
    }

    let mut forward = left.clone();
    forward.merge_from(&right).unwrap();
    let mut backward = right.clone();
    backward.merge_from(&left).unwrap();

    // Identical seed and geometry make the merged bits exactly reproducible.
    assert_eq!(forward.fill_ratio(), single.fill_ratio());
    assert_eq!(forward.fill_ratio(), backward.fill_ratio());
    for i in (0..30_000_u64).step_by(7) {
        assert_eq!(forward.contains_item(&i), single.contains_item(&i));
        assert_eq!(forward.contains_item(&i), backward.contains_item(&i));
    }
}

#[test]
fn counting_bloom_merge_matches_single_filter() {
    let geometry = BloomGeometry {
        bits: 1 << 14,
        hashes: 5,
    };
    let mut left = CountingBloomFilter::from_geometry(geometry, 4);
    let mut right = CountingBloomFilter::from_geometry(geometry, 4);
    let mut single = CountingBloomFilter::from_geometry(geometry, 4);

    for i in 0..2_000_u64 {
        left.insert_item(&i);
        single.insert_item(&i);
    }
    for i in 1_000..3_000_u64 {
        right.insert_item(&i);
        single.insert_item(&i);
    }

    left.merge_from(&right).unwrap();
    assert_eq!(left.fill_ratio(), single.fill_ratio());
    for i in (0..4_000_u64).step_by(11) {
        assert_eq!(left.contains_item(&i), single.contains_item(&i));
    }
}

#[test]
fn count_min_merge_matches_single_sketch_and_commutes() {
    let geometry = CountMinGeometry {
        width: 512,
        depth: 4,
    };
    let counts = stream_counts(1_000);
    let mut left = CountMinSketch::from_geometry(geometry, 21);
    let mut right = CountMinSketch::from_geometry(geometry, 21);
    let mut single = CountMinSketch::from_geometry(geometry, 21);

    for (item, &count) in counts.iter().enumerate() {
        let target = if item % 2 == 0 { &mut left } else { &mut right };
        for _ in 0..count {
            target.insert_item(&(item as u64));
            single.insert_item(&(item as u64));
        }
    }

    let mut forward = left.clone();
    forward.merge_from(&right).unwrap();
    let mut backward = right.clone();
    backward.merge_from(&left).unwrap();

    assert_eq!(forward.total_count(), single.total_count());
    assert_eq!(forward.total_count(), backward.total_count());
    for item in 0..counts.len() as u64 {
        assert_eq!(forward.estimate_item(&item), single.estimate_item(&item));
        assert_eq!(forward.estimate_item(&item), backward.estimate_item(&item));
    }
}

#[test]
fn count_sketch_merge_matches_single_sketch() {
    let geometry = CountSketchGeometry {
        width: 512,
        depth: 5,
    };
    let counts = stream_counts(500);
    let mut left = CountSketch::from_geometry(geometry, 31);
    let mut right = CountSketch::from_geometry(geometry, 31);
    let mut single = CountSketch::from_geometry(geometry, 31);

    for (item, &count) in counts.iter().enumerate() {
        let target = if item % 2 == 0 { &mut left } else { &mut right };
        for _ in 0..count {
            target.insert_item(&(item as u64));
            single.insert_item(&(item as u64));
        }
    }

    left.merge_from(&right).unwrap();
    assert_eq!(left.total_count(), single.total_count());
    for item in 0..counts.len() as u64 {
        assert_eq!(
            left.estimate_signed(&item),
            single.estimate_signed(&item),
            "merged estimate diverged for item {item}"
        );
    }
}

#[test]
fn blocked_bloom_merge_matches_single_filter() {
    use adumbratio::sketch::BlockedBloomFilter;

    let geometry = BloomGeometry {
        bits: 1 << 16,
        hashes: 7,
    };
    let mut left = BlockedBloomFilter::from_geometry(geometry, 9);
    let mut right = BlockedBloomFilter::from_geometry(geometry, 9);
    let mut single = BlockedBloomFilter::from_geometry(geometry, 9);

    for i in 0..10_000_u64 {
        left.insert_item(&i);
        single.insert_item(&i);
    }
    for i in 10_000..20_000_u64 {
        right.insert_item(&i);
        single.insert_item(&i);
    }

    left.merge_from(&right).unwrap();
    assert_eq!(left.fill_ratio(), single.fill_ratio());
    for i in (0..30_000_u64).step_by(7) {
        assert_eq!(left.contains_item(&i), single.contains_item(&i));
    }
}

#[test]
fn hyperloglog_merge_matches_single_sketch_and_commutes() {
    use adumbratio::sketch::HyperLogLog;

    let mut left = HyperLogLog::with_seed(12, 9);
    let mut right = HyperLogLog::with_seed(12, 9);
    let mut single = HyperLogLog::with_seed(12, 9);

    for i in 0..50_000_u64 {
        left.insert_item(&i);
        single.insert_item(&i);
    }
    for i in 50_000..100_000_u64 {
        right.insert_item(&i);
        single.insert_item(&i);
    }

    let mut forward = left.clone();
    forward.merge_from(&right).unwrap();
    let mut backward = right.clone();
    backward.merge_from(&left).unwrap();

    // Register-wise max is deterministic, so merged estimates are identical.
    assert_eq!(forward.cardinality(), single.cardinality());
    assert_eq!(forward.cardinality(), backward.cardinality());
}

#[test]
fn minhash_merge_matches_single_sketch() {
    use adumbratio::sketch::MinHash;

    let mut left = MinHash::with_seed(256, 9);
    let mut right = MinHash::with_seed(256, 9);
    let mut single = MinHash::with_seed(256, 9);

    for i in 0..10_000_u64 {
        left.insert_item(&i);
        single.insert_item(&i);
    }
    for i in 10_000..20_000_u64 {
        right.insert_item(&i);
        single.insert_item(&i);
    }

    left.merge_from(&right).unwrap();
    // Element-wise min is deterministic: the merged signature is identical
    // to one built over the union.
    assert_eq!(left.signature(), single.signature());
    assert_eq!(left.jaccard(&single), 1.0);
}

#[test]
fn theta_merge_matches_single_sketch_and_commutes() {
    use adumbratio::sketch::ThetaSketch;

    let mut left = ThetaSketch::with_seed(256, 9);
    let mut right = ThetaSketch::with_seed(256, 9);
    let mut single = ThetaSketch::with_seed(256, 9);

    for i in 0..20_000_u64 {
        left.insert_item(&i);
        single.insert_item(&i);
    }
    for i in 20_000..40_000_u64 {
        right.insert_item(&i);
        single.insert_item(&i);
    }

    let mut forward = left.clone();
    forward.merge_from(&right).unwrap();
    let mut backward = right.clone();
    backward.merge_from(&left).unwrap();

    // Bottom-k of the union is deterministic: retained lists are identical.
    assert_eq!(forward.retained(), single.retained());
    assert_eq!(forward.retained(), backward.retained());
}

#[test]
fn merge_rejects_incompatible_sketches() {
    let geometry = CountMinGeometry {
        width: 128,
        depth: 4,
    };
    let mut left = CountMinSketch::from_geometry(geometry, 1);
    let other_seed = CountMinSketch::from_geometry(geometry, 2);
    assert_eq!(
        left.merge_from(&other_seed),
        Err(MergeError::SeedMismatch)
    );

    let other_geometry = CountMinSketch::from_geometry(
        CountMinGeometry {
            width: 256,
            depth: 4,
        },
        1,
    );
    assert_eq!(
        left.merge_from(&other_geometry),
        Err(MergeError::GeometryMismatch)
    );

    let bloom_geometry = BloomGeometry {
        bits: 1 << 12,
        hashes: 3,
    };
    let mut bloom = CountingBloomFilter::from_geometry(bloom_geometry, 1);
    let other_bloom = CountingBloomFilter::from_geometry(
        BloomGeometry {
            bits: 1 << 13,
            hashes: 3,
        },
        1,
    );
    assert_eq!(
        bloom.merge_from(&other_bloom),
        Err(MergeError::GeometryMismatch)
    );
}
