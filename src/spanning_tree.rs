//! Functions for generating random spanning trees.
use crate::buffers::SpanningTreeBuffer;
use crate::graph::{Edge, Graph};
use rand::rngs::SmallRng;
use rand::Rng;
use std::cmp::{max, min};

pub trait SpanningTreeSampler {
    /// Samples a random tree of `graph` using `rng`; inserts the tree into `buf`.
    fn random_spanning_tree(
        &mut self,
        graph: &Graph,
        buf: &mut SpanningTreeBuffer,
        rng: &mut SmallRng,
    );
}
pub use crate::spanning_tree::rmst::{RMSTSampler, RegionAwareSampler};
pub use crate::spanning_tree::ust::USTSampler;

/// Spanning tree sampling from the uniform distribution.
mod ust {
    use super::*;
    use crate::buffers::RandomRangeBuffer;

    /// A reusable buffer for Wilson's algorithm.
    pub struct USTBuffer {
        /// Boolean representation of the subset of nodes in the spanning tree.
        pub in_tree: Vec<bool>,
        /// The next node in the spanning tree (for a chosen ordering).
        pub next: Vec<i64>,
        /// The edges in the MST.
        pub edges: Vec<usize>,
    }

    impl USTBuffer {
        /// Creates a buffer for a spanning tree of a subgraph
        /// within a graph of size `n`.
        pub fn new(n: usize) -> USTBuffer {
            return USTBuffer {
                in_tree: vec![false; n],
                next: vec![-1 as i64; n],
                edges: Vec::<usize>::with_capacity(n - 1),
            };
        }

        /// Resets the buffer.
        pub fn clear(&mut self) {
            self.in_tree.fill(false);
            self.next.fill(-1);
            self.edges.clear();
        }
    }

    /// Samples random spanning trees from the uniform distribution.
    pub struct USTSampler {
        /// A buffer for Wilson's algorithm.
        ust_buf: USTBuffer,
        /// A reservoir of random bytes (used for quickly selecting random node neighbors).
        range_buf: RandomRangeBuffer,
    }

    impl USTSampler {
        /// Creates a UST sampler (and underlying buffers) for a graph of approximate
        /// size `n`. (A reservoir of random bytes is initialized using `rng`.)
        pub fn new(n: usize, rng: &mut SmallRng) -> USTSampler {
            USTSampler {
                ust_buf: USTBuffer::new(n),
                range_buf: RandomRangeBuffer::new(rng),
            }
        }
    }

    impl SpanningTreeSampler for USTSampler {
        /// Draws a random spanning tree of a graph from the uniform distribution.
        /// Returns nothing; The MST buffer `buf` is updated in place.
        ///
        /// We use Wilson's algorithm [1] (which is, in essence, a self-avoiding random
        /// walk) to generate the tree.
        ///
        /// # Arguments
        /// * `graph` - The graph to form a spanning tree from. The maximum degree
        ///   of the graph must be ≤256; otherwise, sampling from the uniform
        ///   distribution is not guaranteed.
        /// * `buf` - The buffer to insert the spanning tree into.
        /// * `rng` - A random number generator (used to select the spanning tree
        ///   root and refresh the random byte reservoir).
        ///
        /// # References
        /// [1]  Wilson, David Bruce. "Generating random spanning trees more quickly
        ///      than the cover time." Proceedings of the twenty-eighth annual ACM
        ///      symposium on Theory of computing. 1996.
        fn random_spanning_tree(
            &mut self,
            graph: &Graph,
            buf: &mut SpanningTreeBuffer,
            rng: &mut SmallRng,
        ) {
            buf.clear();
            self.ust_buf.clear();
            let n = graph.pops.len();
            let root = rng.gen_range(0..n);
            self.ust_buf.in_tree[root] = true;
            for i in 0..n {
                let mut u = i;
                while !self.ust_buf.in_tree[u] {
                    let neighbors = &graph.neighbors[u];
                    let neighbor =
                        neighbors[self.range_buf.range(rng, neighbors.len() as u8) as usize];
                    self.ust_buf.next[u] = neighbor as i64;
                    u = neighbor;
                }
                u = i;
                while !self.ust_buf.in_tree[u] {
                    self.ust_buf.in_tree[u] = true;
                    u = self.ust_buf.next[u] as usize;
                }
            }

            for (curr, &prev) in self.ust_buf.next.iter().enumerate() {
                if prev >= 0 {
                    let a = min(curr, prev as usize);
                    let b = max(curr, prev as usize);
                    let mut edge_idx = graph.edges_start[a];
                    while graph.edges[edge_idx].0 == a {
                        if graph.edges[edge_idx].1 == b {
                            self.ust_buf.edges.push(edge_idx);
                            break;
                        }
                        edge_idx += 1;
                    }
                }
            }
            if self.ust_buf.edges.len() != n - 1 {
                panic!(
                    "expected to have {} edges in MST but got {}",
                    n - 1,
                    self.ust_buf.edges.len()
                );
            }

            for &edge in self.ust_buf.edges.iter() {
                let Edge(src, dst) = graph.edges[edge];
                buf.st[src].push(dst);
                buf.st[dst].push(src);
            }
        }
    }
}

/// Spanning tree sampling via random edge weights.
mod rmst {
    use super::*;
    use petgraph::unionfind::UnionFind;
    use rand::seq::SliceRandom;

    /// Samples random spanning trees by sampling random edge weights and finding
    /// the minimum spanning tree.
    pub struct RMSTSampler {
        /// Buffer for randomly ordered edges.
        edges_by_weight: Vec<Edge>,
    }

    /// Samples random spanning trees by sampling random edge weights (with
    /// weight distributions varying based on edge type) and finding the
    /// maximum spanning tree.
    pub struct RegionAwareSampler {
        /// Buffer for random edge weights.
        weights: Vec<f64>,
        /// Buffer for random edge weights with indices.
        weights_with_indices: Vec<(usize, f64)>,
        /// Buffer for randomly ordered edges.
        edges_by_weight: Vec<Edge>,
        /// Sampler configuration (column -> weight).
        region_weights: Vec<(String, f64)>,
    }

    impl RMSTSampler {
        /// Initializes a random MST sampler for a graph with approximate size `n`.
        pub fn new(n: usize) -> RMSTSampler {
            RMSTSampler {
                edges_by_weight: Vec::<Edge>::with_capacity(8 * n),
            }
        }
    }

    impl RegionAwareSampler {
        /// Initializes a random MST sampler for a graph with approximate size `n`.
        pub fn new(n: usize, region_weights: Vec<(String, f64)>) -> RegionAwareSampler {
            RegionAwareSampler {
                weights: Vec::<f64>::with_capacity(8 * n),
                weights_with_indices: Vec::<(usize, f64)>::with_capacity(8 * n),
                edges_by_weight: Vec::<Edge>::with_capacity(8 * n),
                region_weights: region_weights,
            }
        }
    }

    /// Given an edge order (`edges_by_weight`), finds the minimum spanning tree of
    /// `graph` using Kruskal's algorithm and inserts the tree into `buf`.
    fn minimum_spanning_tree(
        graph: &Graph,
        buf: &mut SpanningTreeBuffer,
        edges_by_weight: &Vec<Edge>,
    ) {
        buf.clear();

        // Initialize a union-find data structure to keep track of connected
        // components of the graph.
        // TODO: buffer this?
        let mut uf = UnionFind::<usize>::new(graph.edges.len());

        // Apply Kruskal's algorithm: add edges until the graph is connected.
        let n_edges = graph.pops.len() - 1;
        let mut n_unions = 0;
        for &Edge(src, dst) in edges_by_weight.iter() {
            if n_unions == n_edges {
                break;
            }
            if !uf.equiv(src, dst) {
                uf.union(src, dst);
                buf.st[src].push(dst);
                buf.st[dst].push(src);
                n_unions += 1;
            }
        }
        if n_unions != n_edges {
            panic!(
                "expected to have {} edges in MST but got {}",
                n_edges, n_unions
            );
        }
    }

    impl SpanningTreeSampler for RMSTSampler {
        /// Draws a random spanning tree of a graph by sampling random edge weights
        /// and finding the minimum spanning tree (using Kruskal's algorithm).
        /// Returns nothing; The MST buffer `buf` is updated in place.
        ///
        /// # Arguments
        /// * `graph` - The graph to form a spanning tree from.
        /// * `buf` - The buffer to insert the spanning tree into.
        /// * `rng` - A random number generator (used to generate random edge weights).
        fn random_spanning_tree(
            &mut self,
            graph: &Graph,
            buf: &mut SpanningTreeBuffer,
            rng: &mut SmallRng,
        ) {
            self.edges_by_weight.clone_from(&graph.edges);
            self.edges_by_weight.shuffle(rng);
            minimum_spanning_tree(graph, buf, &self.edges_by_weight);
        }
    }

    impl SpanningTreeSampler for RegionAwareSampler {
        /// Draws a random spanning tree of a graph by sampling random edge weights
        /// and finding the minimum spanning tree (using Kruskal's algorithm).
        /// Reweights edges based on the region-aware settings in the buffer
        /// configuration: edges that span two units (e.g. two counties)
        /// are downweighted by the unit's weight, such that (assuming positive weights)
        /// the minimum spanning tree is more likely to contain edges _between_ units.
        ///  
        /// Returns nothing; The MST buffer `buf` is updated in place.
        ///
        /// # Arguments
        /// * `graph` - The graph to form a spanning tree from.
        /// * `buf` - The buffer to insert the spanning tree into.
        /// * `rng` - A random number generator (used to generate random edge weights).
        fn random_spanning_tree(
            &mut self,
            graph: &Graph,
            buf: &mut SpanningTreeBuffer,
            rng: &mut SmallRng,
        ) {
            // An allocation-free scheme for weight sampling: maintain three
            // separate buffers.
            //   * Efficiently generate raw weights by edge index
            //     using the `weights` buffer. (RNG sampling + offsets)
            //   * Convert these weights to (edge index, edge weight) pairs in
            //     the `weights_with_indices` buffer. Sort this buffer in place.
            //   * Copy edges from the graph into the `edges_by_weight` based
            //     on the edge index order in `weights_with_indices`.
            let n_edges = graph.edges.len();
            self.weights.resize(n_edges, 0.0);
            rng.fill(&mut self.weights[..]);
            for (region_col, region_weight) in self.region_weights.iter() {
                for (idx, edge) in graph.edges.iter().enumerate() {
                    if graph.attr[region_col][edge.0] != graph.attr[region_col][edge.1] {
                        self.weights[idx] -= region_weight;
                    }
                }
            }

            self.weights_with_indices.clear();
            for (idx, &weight) in self.weights.iter().enumerate() {
                self.weights_with_indices.push((idx, weight));
            }
            self.weights_with_indices
                .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().reverse());

            self.edges_by_weight.clear();
            for (edge_idx, _) in self.weights_with_indices.iter() {
                self.edges_by_weight.push(graph.edges[*edge_idx]);
            }

            minimum_spanning_tree(graph, buf, &self.edges_by_weight);
        }
    }
}
