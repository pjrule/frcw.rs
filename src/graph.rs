//! A lightweight graph with population metadata.

/// Edges are pairs of node indices.
#[derive(Clone, Hash, Eq, PartialEq)]
pub struct Edge(pub usize, pub usize);

/// A lightweight graph with population metadata.
#[derive(Clone)]
pub struct Graph {
    /// The graph's edges, represented as pairs of node indices,
    /// sorted by the first element of the pair.
    /// (Nodes are represented implicitly.)
    pub edges: Vec<Edge>,
    /// The population at each node.
    pub pops: Vec<u32>,
    /// The graph's adjacencies (list-of-lists format).
    pub neighbors: Vec<Vec<usize>>,
    /// Maps between node indices and blocks of edges in `edges`.
    /// The nth element corresponds to the starting index of the
    /// block of edges in `edges` of the form (n, *).
    pub edges_start: Vec<usize>,
    /// The total population over all nodes.
    /// (Should be equal to the sum of `pops`.)
    pub total_pop: u32,
}

impl Graph {
    /// Returns a new graph with preallocated containers for `n` nodes and
    /// `8 * n` edges.
    pub fn new_buffer(n: usize) -> Graph {
        return Graph {
            pops: Vec::<u32>::with_capacity(n),
            neighbors: vec![Vec::<usize>::with_capacity(8); n],
            edges: Vec::<Edge>::with_capacity(8 * n),
            edges_start: vec![0 as usize; n],
            total_pop: 0,
        };
    }

    /// Resets a graph's containers.
    /// (Useful when using a graph as a subgraph buffer.)
    pub fn clear(&mut self) {
        self.pops.clear();
        for adj in self.neighbors.iter_mut() {
            adj.clear();
        }
        self.edges.clear();

        // TODO: These technically shouldn't have to be cleared.
        // However, not clearing them explictly could make debugging harder;
        // thus, we leave them in for now.
        self.edges_start.fill(0);
        self.total_pop = 0;
    }
}
