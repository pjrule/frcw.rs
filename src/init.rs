//! Utility functions for loading graph and partition data.
use crate::graph::{Edge, Graph};
use crate::partition::Partition;
use serde_json::Result as SerdeResult;
use serde_json::Value;
use std::fs;

/// Loads graph and partition data in the NetworkX `adjacency_data` format
/// used by [GerryChain](https://github.com/mggg/gerrychain). Returns a
/// [serde_json::Result] containing a [graph::Graph] and
/// a [partition::Partition] upon a successful load.
///
/// # Arguments
///
/// * `path` - the path of the graph JSON file.
/// * `population` - The column in the graph JSON corresponding to total node
///    population. This column should be integer-valued.
/// * `assignment_col` - A column in the graph JSON corresponding to a
///    a seed partition. This column should be integer-valued and 1-indexed.
pub fn from_networkx(
    path: &str,
    pop_col: &str,
    assignment_col: &str,
) -> SerdeResult<(Graph, Partition)> {
    // TODO: should load from a generic buffer.
    let raw = fs::read_to_string(path).expect("Could not load graph");
    let data: Value = serde_json::from_str(&raw)?;

    let raw_nodes = data["nodes"].as_array().unwrap();
    let raw_adj = data["adjacency"].as_array().unwrap();
    let num_nodes = raw_nodes.len();
    let mut pops = Vec::<u32>::with_capacity(num_nodes);
    let mut neighbors = Vec::<Vec<usize>>::with_capacity(num_nodes);
    let mut assignments = Vec::<u32>::with_capacity(num_nodes);
    let mut edges = Vec::<Edge>::new();
    let mut edges_start = vec![0 as usize; num_nodes];

    for (index, (node, adj)) in raw_nodes.iter().zip(raw_adj.iter()).enumerate() {
        edges_start[index] = edges.len();
        let node_neighbors: Vec<usize> = adj
            .as_array()
            .unwrap()
            .into_iter()
            .map(|n| n.as_object().unwrap()["id"].as_u64().unwrap() as usize)
            .collect();
        pops.push(node[pop_col].as_u64().unwrap() as u32);
        neighbors.push(node_neighbors.clone());
        // TODO: we assume that assignments are 1-indexed (and in the range 1..<# of
        // districts>) and convert them to be 0-indexed.  This is not always the case
        // in real data and can be inconvenient to fix.  We should remove this assumption.
        assignments.push((node[assignment_col].as_u64().unwrap() - 1) as u32);

        for neighbor in &node_neighbors {
            if neighbor > &index {
                let edge = Edge(index, *neighbor);
                edges.push(edge.clone());
            }
        }
    }

    let total_pop = pops.iter().sum();
    let num_dists = assignments.iter().max().unwrap() + 1;
    let mut dist_nodes: Vec<Vec<usize>> = (0..num_dists).map(|_| Vec::<usize>::new()).collect();
    for (index, assignment) in assignments.iter().enumerate() {
        assert!(assignment < &num_dists);
        dist_nodes[*assignment as usize].push(index);
    }
    let mut dist_adj = vec![0 as u32; (num_dists * num_dists) as usize];
    let mut cut_edges = Vec::<usize>::new();
    for (index, edge) in edges.iter().enumerate() {
        let dist_a = assignments[edge.0 as usize];
        let dist_b = assignments[edge.1 as usize];
        assert!(dist_a < num_dists);
        assert!(dist_b < num_dists);
        if dist_a != dist_b {
            dist_adj[((dist_a * num_dists) + dist_b) as usize] += 1;
            dist_adj[((dist_b * num_dists) + dist_a) as usize] += 1;
            cut_edges.push(index);
        }
    }
    let mut dist_pops = vec![0 as u32; num_dists as usize];
    for (index, pop) in pops.iter().enumerate() {
        dist_pops[assignments[index] as usize] += pop;
    }

    let graph = Graph {
        pops: pops,
        neighbors: neighbors,
        edges: edges.clone(),
        edges_start: edges_start.clone(),
        total_pop: total_pop,
    };
    let partition = Partition {
        num_dists: num_dists,
        assignments: assignments,
        cut_edges: cut_edges,
        dist_adj: dist_adj,
        dist_pops: dist_pops,
        dist_nodes: dist_nodes,
    };
    return Ok((graph, partition));
}
