# adumbratio

> **Note:** developed with heavy LLM assistance — substantially
> machine-generated with human review. **Use at your own risk:** run
> `cargo test` and the `examples/` against your own workloads before
> trusting it in production.

Composable building blocks for probabilistic data structures ("sketches")
in Rust — plus 25 classical sketches assembled from them.

Most sketch libraries ship each structure as a sealed black box, but
structurally they are all the same few parts in different arrangements:

```
sketch = storage layout × index derivation × update policy
```

`adumbratio` makes those parts first-class. The built-in sketches use only
the public block API, so anything the crate can't compose is a bug in the
blocks, not a special case:

```
┌──────────────────────────────────────────────────────────────────┐
│ sketches   membership: Bloom · CountingBloom · BlockedBloom ·    │
│            Cuckoo · SemiSortedCuckoo · Quotient · Xor ·          │
│            BinaryFuse                                            │
│            frequency: CountMin(±CU) · CountSketch · AMS ·        │
│            MisraGries · SpaceSaving · TopK                       │
│            cardinality: HyperLogLog(+sparse) · Theta             │
│            similarity: MinHash · BBitMinHash · SimHash           │
│            quantiles: KLL · DDSketch      sets: IBLT             │
├──────────────────────────────────────────────────────────────────┤
│ policies   Saturating · Checked · ConservativeUpdate · KickLoop  │
├──────────────────────────────────────────────────────────────────┤
│ hashing    DoubleHashing · EnhancedDoubleHashing · Partitioned · │
│            Blocked · PartialKeyCuckoo                            │
├──────────────────────────────────────────────────────────────────┤
│ storage    BitArray · PackedArray<BITS> · BucketArray · Matrix   │
└──────────────────────────────────────────────────────────────────┘
```

## Quick start

```toml
[dependencies]
adumbratio = "0.1"
```

```rust
use adumbratio::sketch::{BloomFilter, CountMinSketch, HyperLogLog};

// 10M expected items, 1% false-positive rate — m and k solved for you
let mut bloom = BloomFilter::with_capacity(10_000_000, 0.01);
bloom.insert_item("alice");
assert!(bloom.contains_item("alice"));

let mut cms = CountMinSketch::with_error(0.001, 0.01);
cms.insert_count("get /index", 1500); // byte-weighted inserts
assert!(cms.estimate_item("get /index") >= 1500);

let mut hll = HyperLogLog::new(12);
hll.insert_item("10.0.0.1");
assert!(hll.cardinality() >= 0.99);
```

Every sketch also speaks capability traits (`Insert`, `Contains`,
`EstimateCount`, `Merge`, …), so downstream code can be generic over
"anything that answers membership queries".

## Sketch catalog

| Sketch | Answers | Built from |
|---|---|---|
| Bloom filter | membership (no false negatives) | `BitArray` + `DoubleHashing` |
| Counting Bloom filter | membership + deletion | `PackedArray<4>` + `DoubleHashing` + `Saturating` |
| Blocked Bloom filter | membership, cache-friendly | `BitArray` + `Blocked` scheme |
| Cuckoo filter | membership + deletion, high load | `BucketArray<F>` + `PartialKeyCuckoo` + `KickLoop` |
| Semi-sorted cuckoo filter | membership + deletion, 12% smaller | combinatorial-rank buckets in `PackedArray<56>` |
| Quotient filter | membership + deletion + merge | `PackedArray<R>` remainders + 3 `BitArray` metadata rows |
| xor filter | static membership, ~1.23·log2(1/ε) bits/item | 3-segment xor table + hypergraph peeling |
| binary fuse filter | static membership, ~1.125·f bits/item | fuse-schedule xor table + hypergraph peeling |
| Count-Min Sketch (±CU) | frequency (never underestimates) | `Matrix<PackedArray<32>>` + row indexing |
| Count Sketch | frequency (unbiased) | `Matrix<PackedArray<64>>` + signed median read |
| AMS sketch | L2 norm, self-join size, Rényi-2 entropy | tug-of-war signed counters + median-of-means |
| Misra–Gries | deterministic heavy hitters (lower bounds) | k counters + decrement-all |
| Space-Saving | deterministic heavy hitters (upper bounds) | k entries + replace-the-minimum |
| TopK | heavy hitters | `CountMinSketch` or `CountSketch` + candidate set |
| HyperLogLog (+ sparse mode) | cardinality (1.04/√m error) | `PackedArray<6>` registers + max update |
| theta sketch | cardinality + union/intersection/difference | bottom-k (k smallest hashes) |
| MinHash | set similarity (1/√k error) | k hash-derivation minima + element-wise min merge |
| b-bit MinHash | set similarity, 8× smaller signatures | low-b bits of minima + collision correction |
| SimHash | cosine similarity | sign pattern of ±1 bit sums (linear merge) |
| EntropySampler | Shannon entropy | priority-sampling slots + frequency oracle |
| KLL | quantiles / rank (~1/k error) | compactor hierarchy (sorted buffer levels) |
| DDSketch | quantiles with relative error ±α | logarithmic buckets, count-per-bucket |
| IBLT | membership + set reconciliation | `(count, key_sum, hash_sum)` cells |

Each sketch's doc comments cite the paper it reproduces (with DOI links).
The `tests/` suites validate those guarantees empirically: error bounds
from the papers, no false negatives under deletion workloads, and merge
algebra.

## Using it

- **Construction.** Every sketch has geometry-explicit constructors
  (`from_geometry`, `from_parts`) and target-driven ones
  (`with_capacity(n, fpp)`, `with_error(eps, delta)`) that solve the
  standard formulas. All take an explicit `u64` seed.
- **Adversarial inputs.** The default `StableHasher` is deterministic and
  platform-stable (what `merge` and `serde` need) but **not** DoS-resistant.
  For attacker-controlled keys, use a random per-boot seed, or swap any
  `BuildHasher` via `from_parts` — trading merge/serde stability for
  hash-flooding resistance per deployment.
- **Features.** `serde` (serialization), `no_std` (+`alloc`), and `libm`
  (float math without `std`) are all opt-in; the crate is
  `#![forbid(unsafe_code)]` with zero mandatory dependencies.

## Examples, tests, benchmarks

- `cargo run --release --example <name>` — 20 examples that reproduce each
  paper's evaluation on synthetic data (FPP curves, error laws, space
  tables, set reconciliation, entropy) with detailed methodology comments.
- `cargo test` — unit, seeded statistical, merge-algebra, hash-quality, and
  serde round-trip suites. `cargo test --no-default-features --features libm`
  covers the `no_std` build.
- `cargo bench` — 22 criterion groups for every sketch, including
  head-to-head comparisons against the `bloomfilter` and `cuckoofilter`
  crates.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option — the usual Rust convention.
