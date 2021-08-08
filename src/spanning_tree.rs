//! Functions for generating random spanning trees.
use crate::buffers::SpanningTreeBuffer;
use crate::graph::{Edge, Graph};
use rand::rngs::SmallRng;
use rand::Rng;
use std::cmp::{max, min};

pub trait SpanningTreeSampler {
    /// Samples a random tree of `graph` using `rng`; inserts the tree into `buf`.
    fn random_spanning_tree(&mut self, graph: &Graph, buf: &mut SpanningTreeBuffer, rng: &mut SmallRng);
}
pub use crate::spanning_tree::ust::USTSampler;
pub use crate::spanning_tree::rmst::RMSTSampler;


/// Spanning tree sampling from the uniform distribution.
mod ust {
    use super::*;
    use crate::buffers::RandomRangeBuffer;

    /// Samples random spanning trees from the uniform distribution.
     pub struct USTSampler {
        /// A reservoir of random bytes (used for quickly selecting random node neighbors).
        range_buf: RandomRangeBuffer,
    }

    impl USTSampler {
        /// Initializes the UST sampler's reservoir of random bytes using `rng`.
        pub fn new(rng: &mut SmallRng) -> USTSampler {
            USTSampler { range_buf: RandomRangeBuffer::new(rng) }
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
        fn random_spanning_tree(&mut self, graph: &Graph, buf: &mut SpanningTreeBuffer, rng: &mut SmallRng) {
            buf.clear();
            let n = graph.pops.len();
            let root = rng.gen_range(0..n);
            buf.in_tree[root] = true;
            for i in 0..n {
                let mut u = i;
                while !buf.in_tree[u] {
                    let neighbors = &graph.neighbors[u];
                    let neighbor = neighbors[self.range_buf.range(rng, neighbors.len() as u8) as usize];
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
                            buf.edges.push(edge_idx);
                            break;
                        }
                        edge_idx += 1;
                    }
                }
            }
            if buf.edges.len() != n - 1 {
                panic!(
                    "expected to have {} edges in MST but got {}",
                    n - 1,
                    buf.edges.len()
                );
            }

            for &edge in buf.edges.iter() {
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
    /// Samples random spanning trees by sampling random edge weights and finding
    /// the minimum spanning tree.
    pub struct RMSTSampler {}

    impl RMSTSampler {
        pub fn new() -> RMSTSampler {
            RMSTSampler {}
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
        fn random_spanning_tree(&mut self, _graph: &Graph, _buf: &mut SpanningTreeBuffer, _rng: &mut SmallRng) {
            panic!("not implemented");
        }
    }
}
