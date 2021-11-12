// Functional tests that verify that the reversible ReCom chain
// approximately targets the spanning tree distribution on small grids.
use frcw::graph::Graph;
use frcw::partition::Partition;
use frcw::recom::run::multi_chain;
use frcw::recom::{RecomParams, RecomProposal, RecomVariant};
use frcw::stats::{SelfLoopCounts, StatsWriter};
use std::collections::HashSet;
use std::io::Result as IOResult;
use std::iter::FromIterator;


use rstest::rstest;
use test_fixtures::{default_fixture, fixture_with_attributes};


/// A writer that records the cut edge count distribution of a chain.
struct DistStatsWriter {
    last_count: u32
}

impl DistStatsWriter {
    fn new() -> DistStatsWriter {
        return DistStatsWriter {
            last_count: 0
        };
    }
}


#[rstest]
fn test_chain_invariants_recom_grid(
    #[values((6, 6))] pop_range: (u32, u32),
    #[values(1, 4)] n_threads: usize,
    #[values(1, 4)] batch_size: usize,
) {

}
