//! Sum statistics over graph attributes.
use crate::graph::Graph;
use crate::partition::Partition;
use crate::recom::RecomProposal;
use std::collections::HashMap;

/// Computes sums over all statistics for all districts in a proposal.
pub fn partition_sums(graph: &Graph, partition: &Partition) -> HashMap<String, Vec<u32>> {
    graph
        .attr
        .iter()
        .map(|(key, _)| (key.clone(), partition_attr_sums(graph, partition, key)))
        .collect()
}

/// Computes sums over a single statistic for all districts in a proposal.
pub fn partition_attr_sums(graph: &Graph, partition: &Partition, attr: &str) -> Vec<u32> {
    let values = graph.attr.get(attr).unwrap();
    // TODO: check this invariant elsewhere.
    assert!(values.len() == graph.neighbors.len());
    partition
        .dist_nodes
        .iter()
        .map(|nodes| nodes.iter().map(|&n| values[n]).sum())
        .collect()
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
