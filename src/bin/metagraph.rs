//! Fast computation of metagraph connectivity.
use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use itertools::Itertools;
use std::collections::HashMap;
use std::io::{self, BufRead};

type Vec2 = Vec<Vec<usize>>;
type CliqueToIdMap = HashMap<Vec<u128>, usize>;

fn main() {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    let mut line = 0;
    let mut clique_id_to_lines = Vec2::new();
    let mut clique_to_clique_id = CliqueToIdMap::new();
    let mut line_to_clique_ids = Vec2::new();

    while let Some(contents) = lines.next() {
        match contents {
            Err(_) => break, // EOF?
            Ok(ref v) => {
                if v == "" {
                    break; // EOF?
                }
            }
        };
        let mut line_clique_ids = vec![];
        let line_raw = contents.unwrap();
        let first_char = line_raw.chars().next().unwrap();
        if first_char < '0' || first_char > '9' {
            // Non-assignment data.
            continue;
        }
        let assignment: Vec<usize> = line_raw
            .split(" ")
            .map(|a| a.parse::<usize>().unwrap())
            .collect();
        if assignment.len() > 128 {
            // It's really only feasible to compute metagraph statistics
            // for small grids (on the order of ≤64 nodes), so we use 128-bit
            // integers to encode district configurations.
            panic!("Dual graphs with >128 nodes are not supported.");
        }
        // assumption: districts are 0-indexed
        let num_parts = *assignment.iter().max().unwrap() + 1;
        let mut dists = vec![0 as u128; num_parts];
        for (idx, &v) in assignment.iter().enumerate() {
            dists[v] |= 1 << idx;
        }
        dists.sort();
        for clique in dists.into_iter().combinations(num_parts - 2) {
            let clique_id = *clique_to_clique_id
                .entry(clique)
                .or_insert(clique_id_to_lines.len());
            if clique_id_to_lines.len() == clique_id {
                clique_id_to_lines.push(vec![]);
            }
            clique_id_to_lines[clique_id].push(line);
            line_clique_ids.push(clique_id);
        }
        line_to_clique_ids.push(line_clique_ids);
        line += 1;
    }
    let component_sizes = clique_component_sizes(&clique_id_to_lines, &line_to_clique_ids);
    println!("{:?}", component_sizes);
}

/// Computes the sizes of metagraph components via DFS.
fn clique_component_sizes(clique_id_to_nodes: &Vec2, node_to_clique_ids: &Vec2) -> Vec<usize> {
    let num_cliques = clique_id_to_nodes.len();
    let num_nodes = node_to_clique_ids.len();
    let mut component_sizes = vec![];
    let mut cliques_visited = vec![false; num_cliques];
    let mut nodes_visited = vec![false; num_nodes];
    let mut nodes_visited_count = 0;
    let mut nodes_last_visited_count = 0;
    let mut next_cliques = Vec::<usize>::new();
    assert!(num_cliques > 0);

    while nodes_visited_count < num_nodes {
        // Find the next unvisited clique.
        // (We don't expect to have to do this very often.)
        for (clique_id, visited) in cliques_visited.iter().enumerate() {
            if !visited {
                next_cliques.push(clique_id);
                break;
            }
        }
        // Find all neighboring cliques by visiting all nodes in the clique.
        while let Some(clique_id) = next_cliques.pop() {
            for &node in clique_id_to_nodes[clique_id].iter() {
                if !nodes_visited[node] {
                    for &node_clique_id in node_to_clique_ids[node].iter() {
                        if !cliques_visited[node_clique_id] {
                            next_cliques.push(node_clique_id);
                            cliques_visited[node_clique_id] = true;
                        }
                    }
                    nodes_visited[node] = true;
                    nodes_visited_count += 1;
                }
            }
            cliques_visited[clique_id] = true;
        }
        // Update the component sizes.
        component_sizes.push(nodes_visited_count - nodes_last_visited_count);
        nodes_last_visited_count = nodes_visited_count;
    }
    component_sizes
}
