# adumbratio

> **Status: Stage 3 + part of stage 4 implemented.** Storage blocks, hashing
> blocks, policies, capability traits, Bloom, Counting Bloom, Count-Min,
> Count Sketch, Cuckoo filter, and Blocked Bloom are working and tested —
> including a seeded statistical suite that validates the error bounds from
> the papers. `no_std` and `serde` are still pending. See
> [DESIGN.md](DESIGN.md) for the full architecture.

Composable building blocks for probabilistic data structures ("sketches") in
Rust — plus the classical sketches assembled from them.

Most sketch libraries ship each structure as a sealed black box: a Bloom
filter here, a Count-Min Sketch there, each with its own storage code, its
own hashing, its own API. But structurally these are all the same few parts
in different arrangements:

```
sketch = storage layout × index derivation × update policy
```

`adumbratio` makes those parts first-class:

```
┌──────────────────────────────────────────────────────────────────┐
│ sketches   Bloom · CountingBloom · BlockedBloom · CountMin ·     │
│            CountSketch · Cuckoo · …                              │
├──────────────────────────────────────────────────────────────────┤
│ policies   Saturating · Checked · ConservativeUpdate · KickLoop  │
├──────────────────────────────────────────────────────────────────┤
│ hashing    DoubleHashing · EnhancedDoubleHashing · Partitioned · │
│            Blocked · PartialKeyCuckoo                            │
├──────────────────────────────────────────────────────────────────┤
│ storage    BitArray · PackedArray<BITS> · BucketArray · Matrix   │
└──────────────────────────────────────────────────────────────────┘
```

Use the ready-made classical sketches, or build directly from the public
blocks when a workload needs a custom composition.

## Implemented sketches

| Sketch | Answers | Built from | Stage |
|---|---|---|---|
| Bloom filter | membership (no false negatives) | `BitArray` + `DoubleHashing` | ✅ 2 |
| Counting Bloom filter | membership + deletion | `PackedArray<4>` + `DoubleHashing` + `Saturating` | ✅ 3 |
| Count-Min Sketch (+ conservative update) | frequency (never underestimates) | `Matrix<PackedArray<32>>` + row indexing | ✅ 3 |
| Count Sketch | frequency (unbiased) | `Matrix<PackedArray<64>>` + signed median read | ✅ 3 |
| Cuckoo filter | membership + deletion, high load factor | `BucketArray<F>` + `PartialKeyCuckoo` + `KickLoop` | ✅ 3 |
| Blocked Bloom filter | membership, cache-friendly | `BitArray` + `Blocked` scheme | ✅ 4 |
| HyperLogLog | cardinality (1.04/√m error) | `PackedArray<6>` registers + max update | ✅ 5 |
| MinHash | set similarity (1/√k error) | k hash-derivation minima + element-wise min merge | ✅ 5 |
| TopK | heavy hitters | `CountMinSketch` + candidate set | ✅ 5 |
| KLL | quantiles / rank (~1/k error) | compactor hierarchy (levels of sorted buffers) | ✅ 6 |
| xor filter | static membership, ~1.23·log2(1/ε) bits/item | 3-segment xor table + hypergraph peeling | ✅ 6 |
| quotient filter | membership + deletion + merge, cache-linear | `PackedArray<R>` remainders + 3 `BitArray` metadata rows | ✅ 6 |
| theta sketch | cardinality + union/intersection/difference | bottom-k (k smallest hashes) | ✅ 7 |
| SimHash | cosine similarity | sign pattern of ±1 bit sums (linear merge) | ✅ 7 |

## Planned sketches

| Sketch | Answers | Stage |
|---|---|---|
| AMS, Misra–Gries, Space-Saving | L2 norm, deterministic heavy hitters | 7 |
| binary fuse / ribbon, b-bit MinHash, DDSketch, IBLT, … | further family breadth | 8 |

Each sketch's doc comments cite the paper it reproduces (with DOI links),
and the integration tests in `tests/` validate the paper's guarantees
empirically: false-positive rates against the analytical bound, frequency
error within `ε·N`, no false negatives under deletion workloads, and merge
algebra.

## Usage

Ready-made, target-driven construction:

```rust
use adumbratio::sketch::{
    BlockedBloomFilter, BloomFilter, CountMinSketch, CountSketch, CountingBloomFilter,
    CuckooFilter,
};

// 10M expected items, 1% false-positive rate — m and k solved for you
let mut bloom = BloomFilter::with_capacity(10_000_000, 0.01);
bloom.insert_item("alice");
assert!(bloom.contains_item("alice"));

// Same, but all probed bits live in one cache-line block
let mut blocked = BlockedBloomFilter::with_capacity(10_000_000, 0.01);
blocked.insert_item("alice");
assert!(blocked.contains_item("alice"));

let mut counting = CountingBloomFilter::with_capacity(10_000, 0.01);
counting.insert_item("alice");
assert!(counting.remove_item("alice"));

let mut cms = CountMinSketch::with_error(0.001, 0.01);
cms.insert_item("get /index");
assert!(cms.estimate_item("get /index") >= 1);

let mut count_sketch = CountSketch::with_error(0.05, 0.01);
count_sketch.insert_item("get /index");
assert!(count_sketch.estimate_item("get /index") >= 1);

let mut cuckoo = CuckooFilter::with_capacity(1_000, 0.01);
cuckoo.insert_item(&42u64)?;
assert!(cuckoo.remove_item(&42u64));
```

The sketches also speak capability traits, so downstream code can be generic
over "anything that answers membership queries":

```rust
use adumbratio::traits::{Contains, Insert};

fn seen<F>(filter: &mut F, item: &str) -> bool
where
    F: Contains<str> + Insert<str>,
{
    if filter.contains(item) {
        true
    } else {
        let _ = filter.insert(item);
        false
    }
}
```

You can also use the implemented blocks directly:

```rust
use adumbratio::block::{BitArray, PackedArray};

let mut bits = BitArray::new(128);
bits.set(7);
assert!(bits.get(7));

let mut counters = PackedArray::<4>::new(16);
counters.set(3, 15);
assert_eq!(counters.get(3), 15);
```

## Examples: reproducing the papers

Each sketch's doc comments cite the paper it implements. The `examples/`
programs go further and reproduce the papers' evaluation methodology on
synthetic data, printing measured values next to the analytical
predictions:

- `cargo run --release --example bloom_fpp` — empirical FPP vs. Bloom's
  formula across `k` (validating double hashing per Kirsch–Mitzenmacher),
  plus the blocked filter's FPP trade-off per Putze–Sanders–Singler.
- `cargo run --release --example cuckoo_load_fpp` — FPP vs. occupancy, max
  load factor (~95%), and the bits-per-item space comparison with Bloom
  filters, per Fan et al. 2014.
- `cargo run --release --example count_min_zipf` — Count-Min error vs. the
  `ε·N` bound on a Zipf stream, the conservative-update improvement
  (Estan–Varghese), and heavy-hitter accuracy vs. Count Sketch.

## Benchmarks

`cargo bench` runs a criterion suite over insert/query/remove/estimate for
every sketch (deterministic inputs, ~2s per measurement). Baselines are kept
under `target/criterion/` so regressions show up as percentage changes on
subsequent runs.

## Design principles

- **Composability first** — the built-in sketches use only the public block
  API; if the blocks can't express a classic sketch, the blocks are wrong.
- **Correct by construction** — each block owns one invariant (saturated
  counters stay sticky, fingerprints are never the empty marker, cuckoo
  alternate-bucket mapping is an involution), so compositions inherit
  correctness.
- **Deterministic and mergeable** — same seed ⇒ bit-identical sketches across
  platforms; `merge` checks geometry/seed compatibility instead of silently
  corrupting.
- **Pay only for what you use** — zero mandatory dependencies; `serde`,
  external hashers, and `rand` are opt-in features; `no_std` (+`alloc`) is a
  design constraint from the start.

## Roadmap

1. ✅ **Design** — [DESIGN.md](DESIGN.md)
2. ✅ Storage/hashing/policy blocks + capability traits + Bloom filter, fully tested
3. ✅ Counting Bloom, Count-Min (+CU), Count Sketch, Cuckoo filter
4. ◐ Blocked Bloom ✅, statistical/merge suites ✅, `serde` ✅, `no_std`(+`alloc`, optional `libm`) ✅; 0.1 release pending
5. HyperLogLog, MinHash, top-k, benchmarks against existing crates

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option — the usual Rust convention.
