//! A lightweight graph with population metadata.
use std::collections::HashMap;

/// Edges are pairs of node indices.
#[derive(Clone, Hash, Eq, PartialEq, Debug)]
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
    /// Additional node attributes (optional).
    pub attr: HashMap<String, Vec<u32>>,
}

impl Graph {
    /// Returns a new graph with preallocated containers for `n` nodes and
    /// `8 * n` edges.
    pub fn new_buffer(n: usize) -> Graph {
        Graph {
            pops: Vec::<u32>::with_capacity(n),
            neighbors: vec![Vec::<usize>::with_capacity(8); n],
            edges: Vec::<Edge>::with_capacity(8 * n),
            edges_start: vec![0 as usize; n],
            total_pop: 0,
            attr: HashMap::new(),
        }
    }

    /// Returns an nXm rectangular grid graph with unit population at each node.
    /// (Useful for testing.)
    pub fn rect_grid(n: usize, m: usize) -> Graph {
        let size = n * m;
        let mut neighbors = Vec::<Vec<usize>>::with_capacity(size);
        let mut edges = Vec::<Edge>::with_capacity(2 * size);
        let mut edges_start = Vec::<usize>::with_capacity(size);
        for col in 0..n {
            for row in 0..m {
                edges_start.push(edges.len());
                let idx = (col * m) + row;
                let mut node_neighbors = Vec::<usize>::with_capacity(4);
                if col > 0 {
                    let west_idx = ((col - 1) * m) + row;
                    node_neighbors.push(west_idx);
                }
                if row > 0 {
                    let south_idx = (col * m) + (row - 1);
                    node_neighbors.push(south_idx);
                }
                if row < m - 1 {
                    let north_idx = (col * m) + (row + 1);
                    node_neighbors.push(north_idx);
                    edges.push(Edge(idx, north_idx));
                }
                if col < n - 1 {
                    let east_idx = ((col + 1) * m) + row;
                    node_neighbors.push(east_idx);
                    edges.push(Edge(idx, east_idx));
                }
                neighbors.push(node_neighbors);
            }
        }
        Graph {
            pops: vec![1 as u32; size],
            neighbors: neighbors,
            edges: edges,
            edges_start: edges_start,
            total_pop: size as u32,
            attr: HashMap::new(),
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rect_grid_1x1() {
        let grid = Graph::rect_grid(1, 1);
        assert_eq!(grid.edges.len(), 0);
        assert_eq!(grid.pops, vec![1 as u32]);
        assert_eq!(grid.neighbors, vec![vec![0 as usize; 0]]);
        assert_eq!(grid.edges_start, vec![0]);
        assert_eq!(grid.total_pop, 1);
        assert_eq!(grid.attr.len(), 0);
    }

    #[test]
    fn test_rect_grid_2x2() {
        /*
         * 1 - 3
         * |   |
         * 0 - 2
         */
        let grid = Graph::rect_grid(2, 2);
        assert_eq!(
            grid.edges,
            vec![Edge(0, 1), Edge(0, 2), Edge(1, 3), Edge(2, 3)]
        );
        assert_eq!(grid.pops, vec![1 as u32; 4]);
        assert_eq!(
            grid.neighbors,
            vec![vec![1, 2], vec![0, 3], vec![0, 3], vec![1, 2]]
        );
        assert_eq!(grid.edges_start, vec![0, 2, 3, 4]);
        assert_eq!(grid.total_pop, 4);
        assert_eq!(grid.attr.len(), 0);
    }

    #[test]
    fn test_rect_grid_3x2() {
        /*
         * 1 - 3 - 5
         * |   |   |
         * 0 - 2 - 4
         */
        let grid = Graph::rect_grid(3, 2);
        assert_eq!(
            grid.edges,
            vec![
                Edge(0, 1),
                Edge(0, 2),
                Edge(1, 3),
                Edge(2, 3),
                Edge(2, 4),
                Edge(3, 5),
                Edge(4, 5)
            ]
        );
        assert_eq!(grid.pops, vec![1 as u32; 6]);
        assert_eq!(
            grid.neighbors,
            vec![
                vec![1, 2],
                vec![0, 3],
                vec![0, 3, 4],
                vec![1, 2, 5],
                vec![2, 5],
                vec![3, 4]
            ]
        );
        assert_eq!(grid.edges_start, vec![0, 2, 3, 5, 6, 7]);
        assert_eq!(grid.total_pop, 6);
        assert_eq!(grid.attr.len(), 0);
    }
}
