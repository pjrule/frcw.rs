//! Sum statistics over graph attributes.
use crate::graph::Graph;
use crate::partition::Partition;
use crate::recom::RecomProposal;
use std::collections::HashMap;

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
