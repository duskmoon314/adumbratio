# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/).

## [0.1.1] — 2026-07-25

Performance-focused patch release with correctness fixes found by a new
differential test suite; no breaking changes.

### Added

- `TopK::insert_count(item, n)`: weighted insertion through the `Estimator`
  backend, so byte-weighted elephant tracking works over both Count-Min and
  Count Sketch backends. Unit insert is `insert_count(item, 1)`.
- Differential churn test suite (`tests/differential.rs`): randomized
  interleaved workloads continuously verified against exact reference
  models for every sketch family.

### Fixed

- `MisraGries::insert_count` debited every counter by the full weight on
  the miss path, breaking the documented `[f − N/(k+1), f]` bound for
  weights above the smallest counter. It now follows the unit-iterated
  rule exactly (debit `min(count, min-counter)`, track the remainder), so
  weighted and iterated inserts coincide.
- `SpaceSaving::error_bound` returned `N/(k+1)`; the true deterministic
  bound for `k` counters is `N/k` (the recorded error is a past minimum
  counter, and the minimum never exceeds `N/k`). Struct docs corrected.
- `Iblt` cell positions are now drawn distinct per key. A repeated cell
  let a key's key/hash sums xor-cancel while doubling its count, creating
  false pure cells under mixed signs — spurious `DecodeError`s at ~10–20%
  of seeds even at 5.6× overprovisioning, versus the paper's ~1.5× rule of
  thumb that now holds. `insert_item` documents that duplicate inserts are
  unsupported (they xor-cancel; decode fails loudly, never wrongly).

### Performance

- `AmsSketch::insert_item`: bucketed per-group updates replace the textbook
  tug-of-war sweep over every counter — O(depth) per insert instead of
  O(depth × width). 3.16 µs → 68 ns (~46×) at the benchmark geometry, same
  estimator.
- `CountMinSketch`/`CountSketch`/`CountingBloomFilter` hot paths no longer
  allocate per op: CMS insert −7%, CountSketch estimate −18% (stack-buffer
  median), CountingBloom remove −22%.
- `QuotientFilter` hot paths no longer allocate per op: cluster decode
  reuses scratch buffers, and membership runs a single bounded pass instead
  of decoding the cluster. Query hit −75%, query miss −58%, insert+remove
  −24%.

## [0.1.0] — 2026-07-22

First public release. `adumbratio` decomposes probabilistic data structures
("sketches") into composable storage, hashing, and policy blocks, and
assembles 25 classical sketches from them — each with paper citations,
statistical validation, a paper-reproduction example, and benchmarks.

### Added

- **Storage blocks** (`block`): `BitArray`, `PackedArray<BITS>`,
  `BucketArray<F>`, `Matrix<A>`.
- **Hashing blocks** (`hash`): `DoubleHashing`, `EnhancedDoubleHashing`,
  `Partitioned`, `Blocked`, `PartialKeyCuckoo`, and the seeded, stable
  `DefaultBuildHasher`/`StableHasher`.
- **Policies** (`policy`): `Saturating`, `Checked`, `PlainUpdate`,
  `ConservativeUpdate`, `KickLoop`, `XorShift64`/`RngLite`.
- **Capability traits** (`traits`): `Sketch`, `Insert`, `Contains`,
  `Remove`, `EstimateCount`, `EstimateCardinality`, `Merge`, `Estimator`.
- **Membership sketches**: `BloomFilter`, `CountingBloomFilter`,
  `BlockedBloomFilter`, `CuckooFilter<F>` (type-level fingerprint width),
  `SemiSortedCuckooFilter` (combinatorial-rank buckets, 56 vs 64 bits),
  `QuotientFilter` (deletion + merge), `XorFilter`, `BinaryFuseFilter`.
- **Frequency sketches**: `CountMinSketch` (plain and conservative update,
  weighted `insert_count`), `CountSketch` (`insert_count`, `insert_signed`),
  `AmsSketch` (F2, L2, Rényi-2 entropy), `MisraGries`, `SpaceSaving`,
  `TopK` (Count-Min or Count Sketch backend), `EntropySampler` (Shannon
  entropy from uniform event samples).
- **Cardinality**: `HyperLogLog` with HLL++ sparse register mode, and
  `ThetaSketch` (union/intersection/difference/Jaccard).
- **Similarity**: `MinHash`, `BBitMinHash`, `SimHash`.
- **Quantiles**: `KllSketch`, `DdSketch` (relative-error quantiles).
- **Sets**: `Iblt` (invertible Bloom lookup table with exact set
  reconciliation).
- **Features**: `serde` (optional, `no_std`-compatible), `no_std` +
  `alloc` with optional `libm` for float math, MSRV 1.95 (stable minus 2),
  `#![forbid(unsafe_code)]`, zero mandatory dependencies.
- **Tooling**: seeded statistical, merge-algebra, hash-quality, and serde
  round-trip test suites; 20 paper-reproduction examples; 22 criterion
  benchmark groups including head-to-head comparisons against the
  `bloomfilter` and `cuckoofilter` crates; CI covering the feature matrix,
  clippy, MSRV, and all test configurations.
