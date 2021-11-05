//! Statistics for Markov chains.

/// Markov chain self-loop statistics.
mod self_loops;
/// Spanning tree count statistics.
/// This module depends on `ndarray` and `ndarray-linalg` (and therefore depends
/// on a working BLAS installation).
#[cfg(feature = "linalg")]
mod spanning_trees;
/// Graph attribute sum statistics.
mod sums;
/// I/O for statistics.
mod writers;

pub use crate::stats::self_loops::{SelfLoopCounts, SelfLoopReason};
#[cfg(feature = "linalg")]
pub use crate::stats::spanning_trees::subgraph_spanning_tree_count;
pub use crate::stats::sums::{partition_attr_sums, partition_sums, proposal_sums};
pub use crate::stats::writers::{
    AssignmentsOnlyWriter, JSONLWriter, PcompressWriter, StatsWriter, TSVWriter,
};
