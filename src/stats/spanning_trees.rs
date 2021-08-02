//! Spanning tree statistics (number of spanning trees in a district).
use crate::graph::Graph;
use ndarray::*; //{arr2, Array2};
use ndarray_linalg::*; // eig::EigVals;

/// The precision of the eigenvalues (and other intermediate values).
type MatEl = f64;

/// Computes the Laplacian matrix of a subgraph induced by a list of nodes.
fn subgraph_laplacian(graph: &Graph, nodes: &Vec<usize>) -> Array2<MatEl> {
    let n = nodes.len();
    let mut in_nodes = vec![false; graph.neighbors.len()];
    for &node in nodes.iter() {
        in_nodes[node] = true;
    }

    let mut lap = Array2::<MatEl>::zeros((n, n));
    for (ii, &outer) in nodes.iter().enumerate() {
        for (jj, &inner) in nodes.iter().enumerate() {
            if ii > jj {
                continue; // symmetry
            } else if ii == jj {
                // Case: diagonal (node degrees).
                let degree = &graph.neighbors[inner]
                    .iter()
                    .filter(|&n| in_nodes[*n])
                    .count();
                lap[[ii, ii]] = *degree as MatEl;
            } else if graph.neighbors[inner].contains(&outer) {
                // Case: adjacent nodes (-1).
                lap[[ii, jj]] = -1 as MatEl;
                lap[[jj, ii]] = -1 as MatEl;
            }
        }
    }
    lap
}

/// Computes the number of spanning trees in the subgraph induced by the list of nodes.
pub fn subgraph_spanning_tree_count(graph: &Graph, nodes: &Vec<usize>) -> MatEl {
    if graph.neighbors.len() == 1 {
        return 1.0; // special case: single node
    }
    let lap = subgraph_laplacian(graph, nodes);
    let ev = lap.clone().eigvals().unwrap();
    let prod = ev
        .iter()
        .map(|&n| n.re)
        .filter(|&n| n > 1e-6)
        .reduce(|a, b| a * b)
        .unwrap();
    (1 as MatEl / nodes.len() as MatEl) * prod as MatEl
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rstest::rstest;

    #[test]
    #[rustfmt::skip]
    fn subgraph_laplacian_2x2() {
        let expected = arr2(
            &[[ 2., -1., -1.,  0.],
              [-1.,  2.,  0., -1.],
              [-1.,  0.,  2., -1.],
              [ 0., -1., -1.,  2.]]);
        let grid = Graph::rect_grid(2, 2);
        assert_eq!(subgraph_laplacian(&grid, &(0..4).collect()), expected);
    }

    #[test]
    #[rustfmt::skip]
    fn subgraph_laplacian_3x2() {
        let expected = arr2(
            &[[ 2., -1., -1.,  0.,  0.,  0.],
              [-1.,  2.,  0., -1.,  0.,  0.],
              [-1.,  0.,  3., -1., -1.,  0.],
              [ 0., -1., -1.,  3.,  0., -1.],
              [ 0.,  0., -1.,  0.,  2., -1.],
              [ 0.,  0.,  0., -1., -1.,  2.]]);
        let grid = Graph::rect_grid(3, 2);
        assert_eq!(subgraph_laplacian(&grid, &(0..6).collect()), expected);
    }

    #[rstest]
    #[rustfmt::skip]
    fn subgraph_spanning_tree_count_square_grids(
        #[values((1, 1.), (2, 4.), (3, 192.), (4, 100352.), (5, 557568000.),
                 (6, 32565539635200.), (7, 19872369301840986112.),
                 (8, 126231322912498539682594816.),
                 (9, 8326627661691818545121844900397056.),
                 (10, 5694319004079097795957215725765328371712000.))]
        size_count: (usize, MatEl)
    ) {
        // Number of spanning trees in an nXn grid: https://oeis.org/A007341
        let n = size_count.0;
        let expected = size_count.1;
        let grid = Graph::rect_grid(n, n);
        let grid_indices = &(0..(n * n)).collect();
        assert_relative_eq!(
            subgraph_spanning_tree_count(&grid, grid_indices),
            expected,
            max_relative = 0.0001);
    }

    // TODO: more tests for subgraph_spanning_tree_count (don't use whole graph)
}
