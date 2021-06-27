//! Functions for generating random spanning trees.
use crate::buffers::{MSTBuffer, RandomRangeBuffer};
use crate::graph::{Edge, Graph};
use rand::rngs::SmallRng;
use rand::Rng;
use std::cmp::{max, min};

/// Draws a random spanning tree of a graph from the uniform distribution.
/// Returns nothing; The MST buffer `buf` is updated in place.
///
/// We use Wilson's algorithm [1] (which is, in essence, a self-avoiding random
/// walk) to generate the tree.
///
/// # Arguments
/// * `graph` - The graph to form a spanning tree from.
/// * `buf` - The buffer to insert the spanning tree into.
/// * `range_buf` - A reservoir of random bytes (used for quickly selecting
///   random node neighbors).
/// * `rng` - A random number generator (used to select the spanning tree
///   root and refresh the random byte reservoir).
///
/// # References
/// [1]  Wilson, David Bruce. "Generating random spanning trees more quickly
///      than the cover time." Proceedings of the twenty-eighth annual ACM
///      symposium on Theory of computing. 1996.
pub fn uniform_random_spanning_tree(
    graph: &Graph,
    buf: &mut MSTBuffer,
    range_buf: &mut RandomRangeBuffer,
    rng: &mut SmallRng,
) {
    buf.clear();
    let n = graph.pops.len();
    let root = rng.gen_range(0..n);
    buf.in_tree[root] = true;
    for i in 0..n {
        let mut u = i;
        while !buf.in_tree[u] {
            let neighbors = &graph.neighbors[u];
            let neighbor = neighbors[range_buf.range(rng, neighbors.len() as u8) as usize];
            buf.next[u] = neighbor as i64;
            u = neighbor;
        }
        u = i;
        while !buf.in_tree[u] {
            buf.in_tree[u] = true;
            u = buf.next[u] as usize;
        }
    }

    for (curr, &prev) in buf.next.iter().enumerate() {
        if prev >= 0 {
            let a = min(curr, prev as usize);
            let b = max(curr, prev as usize);
            let mut edge_idx = graph.edges_start[a];
            while graph.edges[edge_idx].0 == a {
                if graph.edges[edge_idx].1 == b {
                    buf.mst_edges.push(edge_idx);
                    break;
                }
                edge_idx += 1;
            }
        }
    }
    if buf.mst_edges.len() != n - 1 {
        panic!(
            "expected to have {} edges in MST but got {}",
            n - 1,
            buf.mst_edges.len()
        );
    }

    for &edge in buf.mst_edges.iter() {
        let Edge(src, dst) = graph.edges[edge];
        buf.mst[src].push(dst);
        buf.mst[dst].push(src);
    }
}
