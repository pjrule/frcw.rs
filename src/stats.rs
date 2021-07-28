use crate::graph::Graph;
use crate::partition::Partition;
use crate::recom::RecomProposal;
use serde::ser::{Serialize, Serializer, SerializeStruct};
use serde_json::json;
use std::collections::HashMap;
use std::io::Result;
use std::ops::Add;


/// Reasons why a self-loop occurred in a Markov chain.
#[derive(Hash, Eq, PartialEq, Copy, Clone)]
pub enum SelfLoopReason {
    /// Drew non-adjacent district pairs.
    NonAdjacent,
    /// Drew a spanning tree with no ε-balance nodes
    /// (and therefore no valid splits).
    NoSplit,
    /// Probabilistic rejection based on seam length
    /// (reversible ReCom only).
    SeamLength,
}

/// Self-loop statistics since the last accepted proposal.
pub struct SelfLoopCounts {
    counts: HashMap<SelfLoopReason, usize>,
}

impl Default for SelfLoopCounts {
    fn default() -> SelfLoopCounts {
        return SelfLoopCounts {
            counts: HashMap::new(),
        }
    }
}

impl Add for SelfLoopCounts {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut union = self.counts.clone();
        for (&reason, count) in other.counts.iter() {
            *union.entry(reason).or_insert(0) += count;
        }
        return SelfLoopCounts {
            counts: union
        }
    }
}

impl SelfLoopCounts {
    /// Increments the self-loop count (with a reason).
    pub fn inc(&mut self, reason: SelfLoopReason) {
        *self.counts.entry(reason).or_insert(0) += 1;
    }

    /// Decrements the self-loop count (with a reason).
    pub fn dec(&mut self, reason: SelfLoopReason) {
        *self.counts.entry(reason).or_insert(0) -= 1;
    }

    /// Returns the self-loop count for a reason.
    pub fn get(&self, reason: SelfLoopReason) -> usize {
        self.counts.get(&reason).map_or(0, |&c| c)
    }

    /// Returns the total self-loop count over all reasons.
    pub fn sum(&self) -> usize {
        self.counts.iter().map(|(_, c)| c).sum()
    }

    /// Retrieves an event from the counts based on an arbitrary ordering
    /// and removes the event from the counts.
    ///
    /// Used for sampling events in multithreaded chains: because `SelfLoopCounts`
    /// doesn't store accepted proposal counts, we often want to draw a random event,
    /// accept a proposal if the event index is over/under a threshold, and self-loop
    /// otherwise.
    pub fn index_and_dec(&mut self, index: usize) -> Option<SelfLoopReason> {
        let mut seen = 0;
        for (&reason, count) in self.counts.iter() {
            if seen <= index && index < seen + count {
                self.dec(reason);
                return Some(reason);
            }
            seen += count;
        }
        None
    }
}

impl Serialize for SelfLoopCounts {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("SelfLoopCounts", self.counts.len())?;
        for (&reason, count) in self.counts.iter() {
            // Use camel-case field names in Serde serialization.
            let key = match reason {
                SelfLoopReason::NonAdjacent => "non_adjacent",
                SelfLoopReason::NoSplit => "no_split",
                SelfLoopReason::SeamLength => "seam_length",
            };
            state.serialize_field(key, count)?;
        }
        state.end()
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
        counts: &SelfLoopCounts,
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
        counts: &SelfLoopCounts,
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
