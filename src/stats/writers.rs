use crate::graph::Graph;
use crate::partition::Partition;
use crate::recom::RecomProposal;
use crate::stats::{
    partition_sums, proposal_sums, subgraph_spanning_tree_count, SelfLoopCounts, SelfLoopReason,
};
use serde_json::json;
use std::io::Result;

/// A standard interface for writing steps and statistics to stdout.
/// TODO: allow direct output to a file (e.g. in Parquet format).
/// TODO: move outside of this module.
pub trait StatsWriter {
    /// Prints data from the initial partition.
    fn init(&mut self, graph: &Graph, partition: &Partition) -> Result<()>;

    /// Prints deltas generated from an accepted proposal
    /// which has been applied to `partition`.
    fn step(
        &mut self,
        step: u64,
        graph: &Graph,
        partition: &Partition,
        proposal: &RecomProposal,
        counts: &SelfLoopCounts,
    ) -> Result<()>;

    /// Cleans up after the last step (useful for testing).
    fn close(&mut self) -> Result<()>;
}

/// Writes chain statistics in TSV (tab-separated values) format.
/// Each step in the chain is a line; no statistics are saved about the
/// initial partition.
pub struct TSVWriter {}

/// Writes assignments in space-delimited format (with step number prefix).
pub struct AssignmentsOnlyWriter {}


/// Writes statistics in JSONL (JSON Lines) format.
pub struct JSONLWriter {
    /// Determines whether node deltas should be saved for each step.
    nodes: bool,
    /// Determines whether to compute spanning tree counts for each step.
    spanning_tree_counts: bool,
}

impl TSVWriter {
    pub fn new() -> TSVWriter {
        return TSVWriter {};
    }
}

impl AssignmentsOnlyWriter {
    pub fn new() -> AssignmentsOnlyWriter {
        return AssignmentsOnlyWriter {};
    }
}

impl JSONLWriter {
    pub fn new(nodes: bool, spanning_tree_counts: bool) -> JSONLWriter {
        return JSONLWriter {
            nodes: nodes,
            spanning_tree_counts: spanning_tree_counts,
        };
    }
}

impl StatsWriter for TSVWriter {
    fn init(&mut self, _graph: &Graph, _partition: &Partition) -> Result<()> {
        // TSV column header.
        print!("step\tnon_adjacent\tno_split\tseam_length\ta_label\tb_label\t");
        println!("a_pop\tb_pop\ta_nodes\tb_nodes");
        Ok(())
    }

    fn step(
        &mut self,
        step: u64,
        _graph: &Graph,
        _partition: &Partition,
        proposal: &RecomProposal,
        counts: &SelfLoopCounts,
    ) -> Result<()> {
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:?}\t{:?}",
            step,
            counts.get(SelfLoopReason::NonAdjacent),
            counts.get(SelfLoopReason::NoSplit),
            counts.get(SelfLoopReason::SeamLength),
            proposal.a_label,
            proposal.b_label,
            proposal.a_pop,
            proposal.b_pop,
            proposal.a_nodes,
            proposal.b_nodes
        );
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

impl StatsWriter for JSONLWriter {
    fn init(&mut self, graph: &Graph, partition: &Partition) -> Result<()> {
        // TSV column header.
        let mut stats = json!({
            "num_dists": partition.num_dists,
            "populations": partition.dist_pops,
            "sums": partition_sums(graph, partition)
        });
        if self.spanning_tree_counts {
            stats.as_object_mut().unwrap().insert(
                "spanning_tree_counts".to_string(),
                partition
                    .dist_nodes
                    .iter()
                    .map(|nodes| subgraph_spanning_tree_count(graph, nodes))
                    .collect(),
            );
        }
        println!("{}", json!({ "init": stats }).to_string());
        Ok(())
    }

    fn step(
        &mut self,
        step: u64,
        graph: &Graph,
        _partition: &Partition,
        proposal: &RecomProposal,
        counts: &SelfLoopCounts,
    ) -> Result<()> {
        let mut step = json!({
            "step": step,
            "dists": (proposal.a_label, proposal.b_label),
            "populations": (proposal.a_pop, proposal.b_pop),
            "sums": proposal_sums(graph, proposal),
            "counts": counts,
        });
        if self.nodes {
            step.as_object_mut().unwrap().insert(
                "nodes".to_string(),
                json!((proposal.a_nodes.clone(), proposal.b_nodes.clone())),
            );
        }
        if self.spanning_tree_counts {
            step.as_object_mut().unwrap().insert(
                "spanning_tree_counts".to_string(),
                json!((
                    subgraph_spanning_tree_count(graph, &proposal.a_nodes),
                    subgraph_spanning_tree_count(graph, &proposal.b_nodes)
                )),
            );
        }
        println!("{}", json!({ "step": step }).to_string());
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

impl StatsWriter for AssignmentsOnlyWriter {
    fn init(&mut self, _graph: &Graph, partition: &Partition) -> Result<()> {
        println!("0,{:?}", partition.assignments);
        Ok(())
    }

    fn step(
        &mut self,
        step: u64,
        _graph: &Graph,
        partition: &Partition,
        _proposal: &RecomProposal,
        _counts: &SelfLoopCounts,
    ) -> Result<()> {
        println!("{},{:?}", step, partition.assignments);
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
