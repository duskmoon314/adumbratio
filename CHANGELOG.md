# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/).

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
