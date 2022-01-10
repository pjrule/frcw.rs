//! A lightweight graph with population metadata.
use std::cmp::{max, min};
use std::collections::HashMap;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use snafu::prelude::*;


/// Edges are pairs of node indices.
#[pyclass]
#[derive(Clone, Copy, Hash, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct Edge(pub usize, pub usize);

#[derive(Debug, Snafu)]
pub enum GraphError {
    #[snafu(display("Empty edge list"))]
    ErrEmptyEdgeList,
    #[snafu(display("Could not parse edge index: {edge_index}"))]
    ErrEdgeIndexParse { edge_index: String},
    #[snafu(display("Invalid line in edge list: {line}"))]
    ErrEdgeListLine { line: String},
    #[snafu(display("Edges must be 0-indexed or 1-indexed, but minimum index is {value}"))]
    ErrEdgeIndexMinimumValue {value: usize},
    #[snafu(display("Duplicate edge: {e0} {e1}"))]
    ErrDuplicateEdge { e0: usize, e1: usize },
    #[snafu(display("Mismatch: edge list has {edge_list_nodes} nodes, population list has {pop_list_nodes} nodes"))]
    ErrNodeLengthMismatch {edge_list_nodes: usize, pop_list_nodes: usize},
    #[snafu(display("Could not parse population value: {pop}"))]
    ErrPopulationParse {pop: String },
}

impl std::convert::From<GraphError> for PyErr {
    fn from(err: GraphError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}

/// A lightweight graph with population metadata.
#[pyclass]
#[derive(Clone)]
pub struct Graph {
    /// The graph's edges, represented as pairs of node indices,
    /// sorted by the first element of the pair.
    /// (Nodes are represented implicitly.)
    #[pyo3(get, set)]
    pub edges: Vec<Edge>,
    /// The population at each node.
    #[pyo3(get, set)]
    pub pops: Vec<u32>,
    /// The graph's adjacencies (list-of-lists format).
    #[pyo3(get, set)]
    pub neighbors: Vec<Vec<usize>>,
    /// Maps between node indices and blocks of edges in `edges`.
    /// The nth element corresponds to the starting index of the
    /// block of edges in `edges` of the form (n, *).
    #[pyo3(get, set)]
    pub edges_start: Vec<usize>,
    /// The total population over all nodes.
    /// (Should be equal to the sum of `pops`.)
    #[pyo3(get, set)]
    pub total_pop: u32,
    /// Additional node attributes (optional).
    #[pyo3(get, set)]
    pub attr: HashMap<String, Vec<u32>>,
}

// functions/methods implemented only for rust
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

    /// Initializes a graph from a newline-delimited edge list format representation.
    ///
    /// Arguments:
    ///   * `edge_list`: The list of edges in the graph. Each line contains
    ///     two indices delimited by a space. Nodes are assumed to be labeled
    ///     0..n-1 or 1..n. No edge ordering is assumed.
    ///   * `populations`: The integer population of each node in the graph,
    ///     in index order.
    ///
    /// If the edge list or populations cannot be parsed, or if the number
    /// of nodes is inconsistent between `edge_lists` and `populations`,
    /// and `Err` is returned. Empty graphs are considered invalid, as are
    /// graphs with duplicate edges.
    ///
    /// The caller is responsible for ensuring the graph is connected
    /// (if that property is desired).
    pub fn from_edge_list(edge_list: &str, populations: &str) -> Result<Graph, GraphError> {
        let mut edges = Vec::<Edge>::new();
        if edge_list.is_empty() {
            return Err(GraphError::ErrEmptyEdgeList);
        }
        for line in edge_list.split('\n') {
            if line.is_empty() {
                continue;
            }
            let indices: Vec<&str> = line.split(' ').collect();
            if indices.len() != 2 {
                return Err(GraphError::ErrEdgeListLine {line: line.into()});
            }
            let src = match indices[0].parse::<usize>() {
                Ok(idx) => idx,
                Err(_) => return Err(GraphError::ErrEdgeIndexParse {edge_index: indices[0].into()}),
            };
            let dst = match indices[1].parse::<usize>() {
                Ok(idx) => idx,
                Err(_) => return Err(GraphError::ErrEdgeIndexParse {edge_index: indices[1].into()}),
            };
            edges.push(Edge(min(src, dst), max(src, dst)));
        }
        let min_index = edges.iter().map(|e| e.0).min().unwrap();
        let max_index = edges.iter().map(|e| e.1).max().unwrap();
        let n;
        if min_index == 0 {
            n = max_index + 1;
        } else if min_index == 1 {
            // Fix 1-indexing.
            n = max_index;
            edges = edges.iter().map(|e| Edge(e.0 - 1, e.1 - 1)).collect();
        } else {
            return Err(GraphError::ErrEdgeIndexMinimumValue {value: min_index});
        }
        edges.sort();
        let mut edges_start = Vec::<usize>::new();
        let mut neighbors = vec![Vec::<usize>::new(); n];
        let mut src = edges[0].0;
        edges_start.push(0);
        for (idx, edge) in edges.iter().enumerate() {
            if edge.0 != src {
                // Handle implicit nodes with no edges.
                for _ in src..edge.0 {
                    edges_start.push(idx);
                }
                src = edge.0;
            }
            if neighbors[edge.0].contains(&edge.1) {
                return Err(GraphError::ErrDuplicateEdge {e0: edge.0 + 1, e1: edge.1 + 1} );
            }
            neighbors[edge.0].push(edge.1);
            neighbors[edge.1].push(edge.0);
        }
        // Handle implicit nodes with no edges.
        for _ in src..n - 1 {
            edges_start.push(edges.len());
        }

        let mut parsed_pops = Vec::<u32>::with_capacity(neighbors.len());
        for pop in populations.replace('\n', "").split(' ') {
            match pop.parse::<u32>() {
                Ok(parsed) => parsed_pops.push(parsed),
                Err(_) => return Err(GraphError::ErrPopulationParse {pop: pop.into()}),
            }
        }
        if parsed_pops.len() != neighbors.len() {
            return Err(GraphError::ErrNodeLengthMismatch {edge_list_nodes: neighbors.len(), pop_list_nodes: parsed_pops.len()});
        }

        Ok(Graph {
            total_pop: parsed_pops.iter().sum(),
            pops: parsed_pops,
            neighbors: neighbors,
            edges: edges,
            edges_start: edges_start,
            attr: HashMap::new(),
        })
    }
}

// functions/methods implemented for rust and for Python
#[pymethods]
impl Graph {
    // This is the Python __new__ method for the wrapped class
    # [new]
    fn new (n: usize) -> PyResult<Self> {
        Ok(Graph::new_buffer(n))
    }

    #[staticmethod]
    #[args(name="from_edge_list")]
    fn from_edge_list_py(edge_list: &str, populations: &str) -> PyResult<Self> {
        return match Graph::from_edge_list(edge_list, populations) {
            Ok(parsed_graph) => Ok(parsed_graph),
            Err(graph_error) => Err(graph_error.into()),
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

pub(crate) fn init_submodule (module: &PyModule) -> PyResult<()>  {
    module.add_class::<Edge>()?;
    module.add_class::<Graph>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_grid_1x1() {
        let grid = Graph::rect_grid(1, 1);
        assert_eq!(grid.edges.len(), 0);
        assert_eq!(grid.pops, vec![1 as u32]);
        assert_eq!(grid.neighbors, vec![vec![0 as usize; 0]]);
        assert_eq!(grid.edges_start, vec![0]);
        assert_eq!(grid.total_pop, 1);
        assert_eq!(grid.attr.len(), 0);
    }

    #[test]
    fn rect_grid_2x2() {
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
    fn rect_grid_3x2() {
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

    #[test]
    fn from_edge_list_3x2() {
        // The same graph used in `test_rect_grid_3x2`, but as a shuffled edge list.
        let edge_list = "6 5\n3 5\n4 6\n1 2\n4 2\n3 4\n1 3";
        let pops = "1 2 3 4 5 6";
        let grid = Graph::from_edge_list(edge_list, pops).unwrap();
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
        assert_eq!(grid.pops, vec![1, 2, 3, 4, 5, 6]);
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
        assert_eq!(grid.total_pop, 21);
        assert_eq!(grid.attr.len(), 0);
    }

    #[test]
    fn from_edge_list_disconnected() {
        let edge_list = "1 2\n3 4";
        let pops = "1 2 3 4";
        let grid = Graph::from_edge_list(edge_list, pops).unwrap();
        assert_eq!(grid.edges, vec![Edge(0, 1), Edge(2, 3)]);
        assert_eq!(grid.pops, vec![1, 2, 3, 4]);
        assert_eq!(grid.edges_start, vec![0, 1, 1, 2]);
        assert_eq!(grid.total_pop, 10);
        assert_eq!(grid.attr.len(), 0);
    }

    #[test]
    #[should_panic(expected = "Duplicate edge: 1 2")]
    fn from_edge_list_duplicate_edge() {
        Graph::from_edge_list("1 2\n1 2", "1 2").unwrap();
    }

    #[test]
    #[should_panic(expected = "Invalid line in edge list: 1,2")]
    fn from_edge_list_invalid_edge_list() {
        Graph::from_edge_list("1,2\n2 3", "1 2").unwrap();
    }

    #[test]
    #[should_panic(expected = "Could not parse edge index: a")]
    fn from_edge_list_invalid_left_edge_index() {
        Graph::from_edge_list("1 2\na 3", "1 2").unwrap();
    }

    #[test]
    #[should_panic(expected = "Could not parse edge index: a")]
    fn from_edge_list_invalid_right_edge_index() {
        Graph::from_edge_list("1 2\n3 a", "1 2").unwrap();
    }

    #[test]
    #[should_panic(expected = "Empty edge list")]
    fn from_edge_list_empty_edge_list() {
        Graph::from_edge_list("", "").unwrap();
    }

    #[test]
    #[should_panic(expected = "Could not parse population value: a")]
    fn from_edge_list_invalid_population_value() {
        Graph::from_edge_list("1 2\n2 3", "1 a").unwrap();
    }

    #[test]
    #[should_panic(expected = "Mismatch: edge list has 3 nodes, population list has 2 nodes")]
    fn from_edge_list_length_mismatch() {
        Graph::from_edge_list("1 2\n2 3", "1 2").unwrap();
    }
}
