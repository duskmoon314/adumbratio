//! Matrix update policies.

/// Plain Count-Min update: increment every addressed row cell.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PlainUpdate;

/// Conservative Count-Min update.
///
/// The sketch first reads every addressed row cell, then increments only cells
/// that currently equal the minimum. This usually reduces overestimation bias,
/// but the query path remains the same as plain Count-Min.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConservativeUpdate;
