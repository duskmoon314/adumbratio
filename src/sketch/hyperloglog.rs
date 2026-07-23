//! HyperLogLog implementation.

use core::convert::Infallible;
use core::hash::{BuildHasher, Hash};

use alloc::vec::Vec;

use crate::block::PackedArray;
use crate::error::MergeError;
#[cfg(any(feature = "std", feature = "libm"))]
use crate::float;
use crate::hash::{DefaultBuildHasher, hash_one};
#[cfg(any(feature = "std", feature = "libm"))]
use crate::traits::EstimateCardinality;
use crate::traits::{Insert, Merge, Sketch};

/// A HyperLogLog sketch for distinct-count (cardinality) estimation.
///
/// Hash each item to a uniform 64-bit value and look at the number of
/// leading zeros: seeing a run of `r` zeros has probability `2^-r`, so the
/// maximum run length across a stream is a statistical probe for `log2` of
/// the distinct count. HyperLogLog splits the stream across `m = 2^b`
/// registers — the top `b` hash bits pick the register, the remaining bits
/// supply the zero run — and averages via the harmonic mean, giving
/// standard error `1.04 / sqrt(m)`.
///
/// ```text
/// insert("x")
///      |
///      v
///   hash("x") = [ b index bits | 64-b run bits ]
///                     |              |
///   register M[index] = max(M[index], leading_zeros(run) + 1)
///
/// cardinality()
///      |
///      v
///   alpha_m * m^2 / sum(2^-M[j])     (harmonic mean)
///   small counts: linear counting over the empty registers
/// ```
///
/// Registers are 6-bit cells in a [`PackedArray`], so `m = 16384` registers
/// cost 12 KiB for a standard error of about 0.8%. Small cardinalities use
/// the paper's linear-counting correction; the large-range correction is
/// unnecessary with 64-bit hashes (HLL++ makes the same observation), so it
/// is deliberately omitted.
///
/// # References
///
/// - Philippe Flajolet, Eric Fusy, Olivier Gandouet, and Frederic Meunier,
///   "HyperLogLog: the analysis of a near-optimal cardinality estimation
///   algorithm", AofA 2007. <https://doi.org/10.46298/dmtcs.3545>
/// - Stefan Heule, Marc Nunkesser, and Alexander Hall, "HyperLogLog in
///   Practice: Algorithmic Engineering of a State of the Art Cardinality
///   Estimation Algorithm", EDBT 2013. <https://doi.org/10.1145/2452376.2452456>
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HyperLogLog<S = DefaultBuildHasher> {
    registers: RegisterFile,
    precision: u32,
    seed_fingerprint: u64,
    hasher: S,
}

/// The register storage: HLL++ sparse pairs while few registers are set,
/// the packed dense array otherwise. The two modes are logically identical
/// — every estimate is the same either way.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
enum RegisterFile {
    /// Sorted (index, rho) pairs in parallel vectors, 3 bytes per non-zero
    /// register. Promoted to dense once a quarter of the registers are set
    /// (the point where sparse stops saving memory).
    Sparse { indices: Vec<u16>, rhos: Vec<u8> },
    /// The packed 6-bit register array.
    Dense(PackedArray<6>),
}

impl RegisterFile {
    fn empty(precision: u32) -> Self {
        Self::Sparse {
            indices: Vec::new(),
            rhos: Vec::new(),
        }
        .promote_if_needed(precision)
    }

    fn promote_if_needed(self, precision: u32) -> Self {
        let m = 1_usize << precision;
        match self {
            Self::Sparse { indices, rhos } if indices.len() > m / 4 => {
                let mut dense = PackedArray::new(m);
                for (index, rho) in indices.into_iter().zip(rhos) {
                    dense.set(index as usize, rho as u64);
                }
                Self::Dense(dense)
            }
            other => other,
        }
    }

    fn get(&self, index: usize) -> u64 {
        match self {
            Self::Sparse { indices, rhos } => match indices.binary_search(&(index as u16)) {
                Ok(position) => rhos[position] as u64,
                Err(_) => 0,
            },
            Self::Dense(packed) => packed.get(index),
        }
    }

    fn set_max(&mut self, index: usize, rho: u64, precision: u32) {
        match self {
            Self::Sparse { indices, rhos } => {
                match indices.binary_search(&(index as u16)) {
                    Ok(position) => {
                        if (rhos[position] as u64) < rho {
                            rhos[position] = rho as u8;
                        }
                    }
                    Err(position) => {
                        indices.insert(position, index as u16);
                        rhos.insert(position, rho as u8);
                    }
                }
                let m = 1_usize << precision;
                if indices.len() > m / 4 {
                    let taken = core::mem::replace(
                        self,
                        Self::Sparse {
                            indices: Vec::new(),
                            rhos: Vec::new(),
                        },
                    );
                    *self = taken.promote_if_needed(precision);
                }
            }
            Self::Dense(packed) => {
                packed.update(index, |current| current.max(rho));
            }
        }
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    fn zeros_and_sum(&self, register_count: usize) -> (usize, f64) {
        match self {
            Self::Sparse { indices, rhos } => {
                let zeros = register_count - indices.len();
                let sum = (register_count - indices.len()) as f64
                    + rhos
                        .iter()
                        .map(|&rho| 1.0 / (1_u64 << rho) as f64)
                        .sum::<f64>();
                (zeros, sum)
            }
            Self::Dense(packed) => {
                let mut zeros = 0_usize;
                let mut sum = 0.0_f64;
                for j in 0..register_count {
                    let register = packed.get(j);
                    if register == 0 {
                        zeros += 1;
                    }
                    sum += 1.0 / (1_u64 << register) as f64;
                }
                (zeros, sum)
            }
        }
    }

    fn storage_bytes(&self) -> usize {
        match self {
            Self::Sparse { indices, rhos: _ } => {
                indices.len() * (size_of::<u16>() + size_of::<u8>())
            }
            Self::Dense(packed) => packed.storage_bytes(),
        }
    }

    fn is_sparse(&self) -> bool {
        matches!(self, Self::Sparse { .. })
    }
}

impl HyperLogLog<DefaultBuildHasher> {
    /// Creates a HyperLogLog with precision `b` and seed zero.
    ///
    /// `m = 2^b` registers are allocated; the standard error is
    /// `1.04 / sqrt(m)`. `b` must be in `4..=18`.
    ///
    /// # Panics
    ///
    /// Panics if `b` is outside `4..=18`.
    pub fn new(precision: u32) -> Self {
        Self::with_seed(precision, 0)
    }

    /// Creates a HyperLogLog with precision `b` and an explicit hash seed.
    ///
    /// # Panics
    ///
    /// Panics if `b` is outside `4..=18`.
    pub fn with_seed(precision: u32, seed: u64) -> Self {
        let hasher = DefaultBuildHasher::new(seed);
        Self::from_parts(precision, hasher.seed_fingerprint(), hasher)
    }
}

impl<S> HyperLogLog<S> {
    /// Creates a HyperLogLog from explicit components.
    ///
    /// # Panics
    ///
    /// Panics if `b` is outside `4..=18`.
    pub fn from_parts(precision: u32, seed_fingerprint: u64, hasher: S) -> Self {
        assert!(
            (4..=18).contains(&precision),
            "HyperLogLog precision must be in 4..=18"
        );
        Self {
            registers: RegisterFile::empty(precision),
            precision,
            seed_fingerprint,
            hasher,
        }
    }

    /// Returns the precision `b`; the sketch holds `2^b` registers.
    pub const fn precision(&self) -> u32 {
        self.precision
    }

    /// Returns the number of registers, `2^b`.
    pub const fn register_count(&self) -> usize {
        1 << self.precision
    }

    /// Returns whether the sketch currently uses the sparse register
    /// representation (HLL++): small cardinalities cost 3 bytes per
    /// non-zero register instead of the full packed array.
    pub fn is_sparse(&self) -> bool {
        self.registers.is_sparse()
    }

    /// Returns the seed fingerprint used by merge compatibility checks.
    pub const fn seed_fingerprint(&self) -> u64 {
        self.seed_fingerprint
    }

    /// Returns the theoretical standard error, `1.04 / sqrt(2^b)`.
    ///
    /// Available with the `std` or `libm` feature.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn standard_error(&self) -> f64 {
        1.04 / float::sqrt(self.register_count() as f64)
    }

    /// Returns the byte length of the register storage. In sparse mode this
    /// is 3 bytes per non-zero register; in dense mode the full packed
    /// array.
    pub fn storage_bytes(&self) -> usize {
        self.registers.storage_bytes()
    }

    /// Clears all registers, returning to the sparse representation.
    pub fn clear(&mut self) {
        self.registers = RegisterFile::empty(self.precision);
    }

    /// Returns the raw harmonic-mean estimate and the empty-register count.
    #[cfg(any(feature = "std", feature = "libm"))]
    fn raw_estimate_and_zeros(&self) -> (f64, usize) {
        let m = self.register_count() as f64;
        let (zeros, sum) = self.registers.zeros_and_sum(self.register_count());
        let alpha = match self.register_count() {
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / m),
        };
        (alpha * m * m / sum, zeros)
    }

    /// Estimates the number of distinct inserted items.
    ///
    /// Applies the paper's linear-counting correction when the raw estimate
    /// is small relative to `m`. The large-range correction is omitted: with
    /// 64-bit hashes it only matters beyond ~6e17 items.
    ///
    /// Available with the `std` or `libm` feature.
    #[cfg(any(feature = "std", feature = "libm"))]
    pub fn cardinality(&self) -> f64 {
        let m = self.register_count() as f64;
        let (raw, zeros) = self.raw_estimate_and_zeros();
        if raw <= 2.5 * m && zeros > 0 {
            m * float::ln(m / zeros as f64)
        } else {
            raw
        }
    }
}

impl<S> HyperLogLog<S>
where
    S: BuildHasher,
{
    /// Inserts `item`, raising its register if the new zero run is longer.
    pub fn insert_item<T>(&mut self, item: &T)
    where
        T: Hash + ?Sized,
    {
        let hash = hash_one(&self.hasher, item);
        let index = (hash >> (64 - self.precision)) as usize;
        let run = hash << self.precision;
        let rho = (run.leading_zeros() + 1).min(64 - self.precision + 1);
        self.registers.set_max(index, rho as u64, self.precision);
    }
}

impl<S> Sketch for HyperLogLog<S> {
    fn clear(&mut self) {
        self.clear();
    }

    fn len_hint(&self) -> Option<u64> {
        #[cfg(any(feature = "std", feature = "libm"))]
        {
            let estimate = self.cardinality();
            if estimate.is_finite() && estimate >= 0.0 && estimate <= u64::MAX as f64 {
                Some(float::round(estimate) as u64)
            } else {
                None
            }
        }
        #[cfg(not(any(feature = "std", feature = "libm")))]
        {
            None
        }
    }

    fn storage_bytes(&self) -> usize {
        self.storage_bytes()
    }
}

impl<T, S> Insert<T> for HyperLogLog<S>
where
    T: Hash + ?Sized,
    S: BuildHasher,
{
    type Err = Infallible;

    fn insert(&mut self, item: &T) -> Result<(), Self::Err> {
        self.insert_item(item);
        Ok(())
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl<S> EstimateCardinality for HyperLogLog<S> {
    fn cardinality(&self) -> f64 {
        self.cardinality()
    }
}

impl<S> Merge for HyperLogLog<S> {
    fn merge_from(&mut self, other: &Self) -> Result<(), MergeError> {
        if self.precision != other.precision {
            return Err(MergeError::GeometryMismatch);
        }
        if self.seed_fingerprint != other.seed_fingerprint {
            return Err(MergeError::SeedMismatch);
        }

        match (&mut self.registers, &other.registers) {
            (
                RegisterFile::Sparse { indices, rhos },
                RegisterFile::Sparse {
                    indices: other_indices,
                    rhos: other_rhos,
                },
            ) => {
                // Sorted zip-merge with max on shared indices.
                let mut merged_indices = Vec::with_capacity(indices.len() + other_indices.len());
                let mut merged_rhos = Vec::with_capacity(indices.len() + other_rhos.len());
                let mut left = indices.iter().zip(rhos.iter()).peekable();
                let mut right = other_indices.iter().zip(other_rhos.iter()).peekable();
                loop {
                    match (left.peek(), right.peek()) {
                        (Some(&(&li, &lr)), Some(&(&ri, &rr))) => {
                            if li == ri {
                                merged_indices.push(li);
                                merged_rhos.push(lr.max(rr));
                                left.next();
                                right.next();
                            } else if li < ri {
                                merged_indices.push(li);
                                merged_rhos.push(lr);
                                left.next();
                            } else {
                                merged_indices.push(ri);
                                merged_rhos.push(rr);
                                right.next();
                            }
                        }
                        (Some(&(&li, &lr)), None) => {
                            merged_indices.push(li);
                            merged_rhos.push(lr);
                            left.next();
                        }
                        (None, Some(&(&ri, &rr))) => {
                            merged_indices.push(ri);
                            merged_rhos.push(rr);
                            right.next();
                        }
                        (None, None) => break,
                    }
                }
                *indices = merged_indices;
                *rhos = merged_rhos;
                let taken = core::mem::replace(
                    &mut self.registers,
                    RegisterFile::Sparse {
                        indices: Vec::new(),
                        rhos: Vec::new(),
                    },
                );
                self.registers = taken.promote_if_needed(self.precision);
            }
            _ => {
                // Either side is dense: convert self to dense and fold the
                // other's registers in with max.
                let taken = core::mem::replace(
                    &mut self.registers,
                    RegisterFile::Sparse {
                        indices: Vec::new(),
                        rhos: Vec::new(),
                    },
                );
                let mut dense = match taken {
                    RegisterFile::Sparse { indices, rhos } => {
                        let mut packed = PackedArray::new(self.register_count());
                        for (index, rho) in indices.into_iter().zip(rhos) {
                            packed.set(index as usize, rho as u64);
                        }
                        packed
                    }
                    RegisterFile::Dense(packed) => packed,
                };
                for index in 0..self.register_count() {
                    let theirs = other.registers.get(index);
                    if theirs > 0 {
                        dense.update(index, |current| current.max(theirs));
                    }
                }
                self.registers = RegisterFile::Dense(dense);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::HyperLogLog;
    use crate::error::MergeError;
    use crate::traits::{EstimateCardinality, Insert, Merge, Sketch};

    #[test]
    fn starts_sparse_and_promotes_past_threshold() {
        let mut sketch = HyperLogLog::with_seed(12, 5);
        assert!(sketch.is_sparse());
        assert!(sketch.storage_bytes() < 100);

        for i in 0..2_000_u64 {
            sketch.insert_item(&i);
        }
        // m / 4 = 1024 registers set past the promotion point.
        assert!(!sketch.is_sparse());
    }

    #[test]
    fn estimates_are_consistent_across_promotion() {
        let mut sketch = HyperLogLog::with_seed(12, 5);
        for i in 0..500_u64 {
            sketch.insert_item(&i);
        }
        let sparse_estimate = EstimateCardinality::cardinality(&sketch);
        assert!(sketch.is_sparse());
        assert!(
            (sparse_estimate - 500.0).abs() / 500.0 <= 4.0 * sketch.standard_error(),
            "sparse estimate {sparse_estimate} off for n = 500"
        );

        for i in 500..2_000_u64 {
            sketch.insert_item(&i);
        }
        assert!(!sketch.is_sparse());
        let dense_estimate = EstimateCardinality::cardinality(&sketch);

        let mut whole = HyperLogLog::with_seed(12, 5);
        for i in 0..2_000_u64 {
            whole.insert_item(&i);
        }
        assert_eq!(
            dense_estimate,
            EstimateCardinality::cardinality(&whole),
            "promotion changed the estimate"
        );
    }

    #[test]
    fn merge_works_across_modes() {
        let mut sparse = HyperLogLog::with_seed(12, 7);
        let mut dense = HyperLogLog::with_seed(12, 7);
        for i in 0..10_u64 {
            sparse.insert_item(&i);
        }
        for i in 0..3_000_u64 {
            dense.insert_item(&i);
        }
        assert!(sparse.is_sparse());
        assert!(!dense.is_sparse());

        sparse.merge_from(&dense).unwrap();
        assert!(!sparse.is_sparse());
        let merged = EstimateCardinality::cardinality(&sparse);
        let direct = EstimateCardinality::cardinality(&dense);
        assert_eq!(merged, direct);
    }

    #[test]
    #[should_panic(expected = "HyperLogLog precision must be in 4..=18")]
    fn precision_below_range_is_rejected() {
        HyperLogLog::new(3);
    }

    #[test]
    #[should_panic(expected = "HyperLogLog precision must be in 4..=18")]
    fn precision_above_range_is_rejected() {
        HyperLogLog::new(19);
    }

    #[test]
    fn empty_sketch_estimates_zero() {
        let sketch = HyperLogLog::new(12);
        assert_eq!(EstimateCardinality::cardinality(&sketch), 0.0);
        assert_eq!(Sketch::len_hint(&sketch), Some(0));
    }

    #[test]
    fn repeated_inserts_do_not_inflate() {
        let mut sketch = HyperLogLog::new(12);
        for _ in 0..1_000 {
            sketch.insert_item("same");
        }
        let estimate = EstimateCardinality::cardinality(&sketch);
        assert!(
            estimate <= 2.0,
            "one distinct item estimated as {estimate}"
        );
    }

    #[test]
    fn standard_error_matches_textbook() {
        let sketch = HyperLogLog::new(12);
        assert!((sketch.standard_error() - 1.04 / 64.0).abs() < 1e-12);
    }

    #[test]
    fn capability_traits_work() {
        let mut sketch = HyperLogLog::new(10);
        Insert::<str>::insert(&mut sketch, "alice").unwrap();
        assert!(Sketch::len_hint(&sketch).unwrap() >= 1);
    }

    #[test]
    fn merge_takes_register_maxima_and_commutes() {
        let mut left = HyperLogLog::with_seed(10, 7);
        let mut right = HyperLogLog::with_seed(10, 7);
        let mut single = HyperLogLog::with_seed(10, 7);

        for i in 0..5_000_u64 {
            left.insert_item(&i);
            single.insert_item(&i);
        }
        for i in 5_000..10_000_u64 {
            right.insert_item(&i);
            single.insert_item(&i);
        }

        left.merge_from(&right).unwrap();
        let merged = EstimateCardinality::cardinality(&left);
        let direct = EstimateCardinality::cardinality(&single);
        assert_eq!(merged, direct);

        let other_precision = HyperLogLog::with_seed(11, 7);
        assert_eq!(
            left.merge_from(&other_precision),
            Err(MergeError::GeometryMismatch)
        );
        let other_seed = HyperLogLog::with_seed(10, 8);
        assert_eq!(left.merge_from(&other_seed), Err(MergeError::SeedMismatch));
    }
}
