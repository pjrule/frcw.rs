//! Buffer data structures to avoid memory reallocation.
pub use self::random_range::RandomRangeBuffer;
pub use self::spanning_tree::SpanningTreeBuffer;
pub use self::split::SplitBuffer;
pub use self::subgraph::SubgraphBuffer;

/// Buffers are intended to be lightweight, reusable containers
/// that improve the efficiency of inner loops. In most buffers,
/// fields are intended to be mutated directly, and invariants
/// are not strictly enforced.

/// Buffer for subgraphs.
mod subgraph {
    use crate::graph::Graph;
    /// A reusable buffer for subgraphs of a [graph::Graph] (the "parent graph").
    pub struct SubgraphBuffer {
        /// The nodes in the subgraph.
        pub raw_nodes: Vec<usize>,
        /// A mapping between node IDs in the parent graph and the indices
        /// of `raw_nodes` (which are node IDs in `graph`). If a node
        /// does not appear in the subgraph, its index is -1.
        pub node_to_idx: Vec<i64>,
        /// A subgraph of the parent graph, with nodes relabeled to have
        /// consecutive node IDs. `nodes_to_idx` is used to map between
        /// node IDs in the subgraph and node IDs in the parent graph.
        pub graph: Graph,
    }

    impl SubgraphBuffer {
        /// Creates a new [SubgraphBuffer] of size `b` for a graph of size `n`.
        pub fn new(n: usize, b: usize) -> SubgraphBuffer {
            return SubgraphBuffer {
                raw_nodes: Vec::<usize>::with_capacity(b),
                node_to_idx: vec![-1 as i64; n],
                graph: Graph::new_buffer(b),
            };
        }

        /// Resets the buffer.
        pub fn clear(&mut self) {
            self.raw_nodes.clear();
            self.node_to_idx.fill(-1);
            self.graph.clear();
        }
    }
}

/// Buffer for spanning trees.
mod spanning_tree {
    /// A reusable spanning tree buffer.
    pub struct SpanningTreeBuffer {
        /// The neighbors of each node in the MST (list-of-lists representation).
        pub st: Vec<Vec<usize>>,
    }

    impl SpanningTreeBuffer {
        /// Creates a buffer for a spanning tree of a subgraph
        /// within a graph of size `n`.
        pub fn new(n: usize) -> SpanningTreeBuffer {
            SpanningTreeBuffer {
                st: vec![Vec::<usize>::with_capacity(8); n],
            }
        }

        /// Resets the buffer.
        pub fn clear(&mut self) {
            for node in self.st.iter_mut() {
                node.clear();
            }
        }
    }
}

/// Buffer for spanning tree splits (used in ReCom).
mod split {
    use std::collections::VecDeque;

    /// A reusable buffer for splits of a spanning tree.
    ///
    /// Finding population-balanced splits (or cuts) of a spanning tree
    /// is a key step in the ReCom algorithm. We can compute the population
    /// split of each cut by performing a BFS of a spanning tree.
    /// Essentially, the BFS orients the tree; this buffer represents the
    /// state of a BFS *and* the resulting nodes that root ε-balanced splits.
    // TODO: this really should be broken up into three buffers: a
    // BFSBuffer, a BalanceNodeBuffer, and a container for `in_a`.
    pub struct SplitBuffer {
        /// Boolean representation of whether a node has been visited in the BFS.
        pub visited: Vec<bool>,
        /// The predecessor of each node in the BFS orientation.
        pub pred: Vec<usize>,
        /// The successors of each node in the BFS orientation.
        pub succ: Vec<Vec<usize>>,
        /// A deque (double-ended queue) used for the BFS and
        /// for finding balance nodes.
        pub deque: VecDeque<usize>,
        /// The populations of the subtrees rooted at each node
        /// in the BFS orientation.
        pub tree_pops: Vec<u32>,
        /// Boolean representation of whether the population of
        /// the subtree rooted at a node (in the BFS orientation)
        /// has been computed.
        pub pop_found: Vec<bool>,
        /// The nodes that root ε-balanced splits.
        pub balance_nodes: Vec<usize>,
        /// Boolean representation of whether a node is in the `a`-half of a split.
        pub in_a: Vec<bool>,
    }

    impl SplitBuffer {
        /// Creates a new split buffer for a graph of size `n`
        /// with a soft upper bound of `m` balance nodes.
        pub fn new(n: usize, m: usize) -> SplitBuffer {
            return SplitBuffer {
                visited: vec![false; n],
                pred: vec![0; n],
                succ: vec![Vec::<usize>::with_capacity(8); n],
                deque: VecDeque::<usize>::with_capacity(n),
                tree_pops: vec![0 as u32; n],
                pop_found: vec![false; n],
                balance_nodes: Vec::<usize>::with_capacity(m),
                in_a: vec![false; n],
            };
        }

        /// Resets the buffer.
        pub fn clear(&mut self) {
            self.visited.fill(false);
            for node in self.succ.iter_mut() {
                node.clear();
            }
            self.pop_found.fill(false);
            self.in_a.fill(false);
            self.balance_nodes.clear();

            // TODO: These technically shouldn't have to be cleared.
            // However, not clearing them explictly could make debugging harder;
            // thus, we leave them in for now.
            self.tree_pops.fill(0);
            self.pred.fill(0);
            self.deque.clear();
        }
    }
}

/// Buffer for random bytes.
mod random_range {
    use rand::rngs::SmallRng;
    use rand::Rng;
    use std::num::Wrapping;

    /// Size of the buffer of random values.
    // (We try to set a buffer size that balances refresh time
    // and average efficiency per sample.)
    const RANGE_BUF_SIZE: usize = 1 << 20;

    /// A buffer used for uniformly sampling bytes.
    ///
    /// Unlike most other buffers, this buffer is intended to be opaque;
    /// values should be sampled with [RandomRangeBuffer::next].
    pub struct RandomRangeBuffer {
        buf: Vec<u8>,
        size: usize,
        pos: usize,
    }

    impl RandomRangeBuffer {
        /// Creates a new buffer for uniformly sampling bytes.
        /// The buffer is prepopulated with `rng`.
        pub fn new(rng: &mut SmallRng) -> RandomRangeBuffer {
            let mut buf = vec![0 as u8; RANGE_BUF_SIZE];
            rng.fill(&mut buf[..]);
            return RandomRangeBuffer {
                buf: buf,
                size: RANGE_BUF_SIZE,
                pos: 0,
            };
        }

        /// Gets the next value from the buffer, refreshing the buffer
        /// using `rng` if necessary.
        fn next(&mut self, rng: &mut SmallRng) -> u8 {
            let val = self.buf[self.pos];
            self.pos += 1;
            if self.pos == self.size {
                rng.fill(&mut self.buf[..]);
                self.pos = 0;
            }
            return val;
        }

        /// Uniformly samples a byte in the range [0, ub), refreshing
        /// the buffer of random values using `rng` if necessary.
        ///
        /// Uniform sampling in an arbitrary range is rather subtle
        /// (for instance, the standard modulus trick is both inefficient
        ///  and biased). For efficiency, we sample single bytes at a time.
        ///
        /// This is (usually) sufficient for the primary inner-loop use
        /// case of this buffer: choosing random neighbors of a node
        /// when generating a random spanning tree using Wilson's algorithm.
        pub fn range(&mut self, rng: &mut SmallRng, ub: u8) -> u8 {
            // https://www.pcg-random.org/posts/bounded-rands.html
            // https://lemire.me/blog/2019/06/06/nearly-divisionless-
            // random-integer-generation-on-various-systems/
            let mut x = self.next(rng);
            let mut m = (x as u16) * (ub as u16);
            let mut l = Wrapping(m).0 as u8;
            if l < ub {
                let t = (Wrapping(0) - Wrapping(ub)).0 % ub;
                while l < t {
                    x = self.next(rng);
                    m = (x as u16) * (ub as u16);
                    l = Wrapping(m).0 as u8;
                }
            }
            return Wrapping(m >> 8).0 as u8;
        }
    }
}
