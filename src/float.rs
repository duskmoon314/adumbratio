//! Internal float-math shim.
//!
//! Geometry solvers and error estimators need logarithms and powers, which
//! `core` does not provide. With `std` (the default) these call the
//! inherent `f64` methods; in `no_std` builds the optional `libm` feature
//! supplies the same functions. Functions using this shim exist only when
//! one of the two is enabled.

macro_rules! unary_shim {
    ($name:ident, $method:ident, $libm_fn:ident) => {
        /// See `f64::$method`.
        #[cfg(feature = "std")]
        pub(crate) fn $name(x: f64) -> f64 {
            x.$method()
        }

        /// See `libm::$libm_fn`.
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        pub(crate) fn $name(x: f64) -> f64 {
            libm::$libm_fn(x)
        }
    };
}

unary_shim!(ln, ln, log);
unary_shim!(log2, log2, log2);
unary_shim!(exp, exp, exp);
unary_shim!(sqrt, sqrt, sqrt);
unary_shim!(cos, cos, cos);
unary_shim!(floor, floor, floor);
unary_shim!(ceil, ceil, ceil);
unary_shim!(round, round, round);

/// See `f64::powf`.
#[cfg(feature = "std")]
pub(crate) fn powf(x: f64, y: f64) -> f64 {
    x.powf(y)
}

/// See `libm::pow`.
#[cfg(all(not(feature = "std"), feature = "libm"))]
pub(crate) fn powf(x: f64, y: f64) -> f64 {
    libm::pow(x, y)
}

/// See `f64::powi`.
#[cfg(feature = "std")]
pub(crate) fn powi(x: f64, n: i32) -> f64 {
    x.powi(n)
}

/// See `libm::pow`.
#[cfg(all(not(feature = "std"), feature = "libm"))]
pub(crate) fn powi(x: f64, n: i32) -> f64 {
    libm::pow(x, f64::from(n))
}
