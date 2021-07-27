use crate::graph::Graph;
use crate::partition::Partition;
use crate::recom::RecomProposal;
use serde::Serialize;
use serde_json::json;
use std::collections::HashMap;
use std::io::Result;
use std::ops::Add;

/// Self-loop statistics since the last accepted proposal.
#[derive(Copy, Clone, Default, Serialize)]
pub struct ChainCounts {
    /// The number of self-loops from non-adjacent district pairs.
    pub non_adjacent: usize,
    /// The number of self-loops from drawing a spanning tree
    /// with no ε-balance nodes (and therefore no valid splits).
    pub no_split: usize,
    /// The number of self-loops from probabilistic rejection based
    /// on seam length (reversible ReCom only).
    pub seam_length: usize,
}

impl Add for ChainCounts {
    type Output = Self;

    /// Adds self-loop statistics.
    fn add(self, other: Self) -> Self {
        Self {
            non_adjacent: self.non_adjacent + other.non_adjacent,
            no_split: self.no_split + other.no_split,
            seam_length: self.seam_length + other.seam_length,
        }
    }
}

/// Computes sums over statistics for all districts in a proposal.
pub fn partition_sums(graph: &Graph, partition: &Partition) -> HashMap<String, Vec<u32>> {
    return graph
        .attr
        .iter()
        .map(|(key, values)| {
            // TODO: check this invariant elsewhere.
            assert!(values.len() == graph.neighbors.len());
            let dist_sums = partition
                .dist_nodes
                .iter()
                .map(|nodes| nodes.iter().map(|&n| values[n]).sum())
                .collect();
            return (key.clone(), dist_sums);
        })
        .collect();
}

/// Computes sums over statistics for the two new districts in a proposal.
pub fn proposal_sums(graph: &Graph, proposal: &RecomProposal) -> HashMap<String, (u32, u32)> {
    return graph
        .attr
        .iter()
        .map(|(key, values)| {
            let a_sum = proposal.a_nodes.iter().map(|&n| values[n]).sum();
            let b_sum = proposal.b_nodes.iter().map(|&n| values[n]).sum();
            return (key.clone(), (a_sum, b_sum));
        })
        .collect();
}

/// A standard interface for writing steps and statistics to stdout.
/// TODO: allow direct output to a file (e.g. in Parquet format).
/// TODO: move outside of this module.
pub trait StatsWriter {
    /// Prints data from the initial partition.
    fn init(&mut self, graph: &Graph, partition: &Partition) -> Result<()>;

    /// Prints deltas generated from an accepted proposal.
    fn step(
        &mut self,
        step: u64,
        graph: &Graph,
        proposal: &RecomProposal,
        counts: &ChainCounts,
    ) -> Result<()>;

    /// Cleans up after the last step (useful for testing).
    fn close(&mut self) -> Result<()>;
}

/// Writes chain statistics in TSV (tab-separated values) format.
/// Each step in the chain is a line; no statistics are saved about the
/// initial partition.
pub struct TSVWriter {}

/// Writes statistics in JSONL (JSON Lines) format.
pub struct JSONLWriter {
    /// Determines whether node deltas should be saved for each step.
    nodes: bool,
}

impl TSVWriter {
    pub fn new() -> TSVWriter {
        return TSVWriter {};
    }
}

impl JSONLWriter {
    pub fn new(nodes: bool) -> JSONLWriter {
        return JSONLWriter { nodes: nodes };
    }
}

impl StatsWriter for TSVWriter {
    fn init(&mut self, _graph: &Graph, _partition: &Partition) -> Result<()> {
        // TSV column header.
        print!("step\tnon_adjacent\tno_split\tseam_length\ta_label\tb_label\t");
        println!("a_pop\tb_pop\ta_nodes\tb_nodes");
        return Ok(());
    }

    fn step(
        &mut self,
        step: u64,
        _graph: &Graph,
        proposal: &RecomProposal,
        counts: &ChainCounts,
    ) -> Result<()> {
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:?}\t{:?}",
            step,
            counts.non_adjacent,
            counts.no_split,
            counts.seam_length,
            proposal.a_label,
            proposal.b_label,
            proposal.a_pop,
            proposal.b_pop,
            proposal.a_nodes,
            proposal.b_nodes
        );
        return Ok(());
    }

    fn close(&mut self) -> Result<()> {
        return Ok(());
    }
}

impl StatsWriter for JSONLWriter {
    fn init(&mut self, graph: &Graph, partition: &Partition) -> Result<()> {
        // TSV column header.
        let stats = json!({
            "init": {
                "num_dists": partition.num_dists,
                "populations": partition.dist_pops,
                "sums": partition_sums(graph, partition)
            }
        });
        println!("{}", stats.to_string());
        return Ok(());
    }

    fn step(
        &mut self,
        step: u64,
        graph: &Graph,
        proposal: &RecomProposal,
        counts: &ChainCounts,
    ) -> Result<()> {
        let mut step = json!({
            "step": step,
            "dists": (proposal.a_label, proposal.b_label),
            "populations": (proposal.a_pop, proposal.b_pop),
            "sums": proposal_sums(graph, proposal),
            "counts": counts
        });
        if self.nodes {
            step.as_object_mut().unwrap().insert(
                "nodes".to_string(),
                json!((proposal.a_nodes.clone(), proposal.b_nodes.clone())),
            );
        }
        println!("{}", json!({ "step": step }).to_string());
        return Ok(());
    }

    fn close(&mut self) -> Result<()> {
        return Ok(());
    }
}
