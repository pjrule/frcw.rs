// Functional tests that verify ReCom chain invariants at each step.
use frcw::graph::Graph;
use frcw::partition::Partition;
use frcw::recom::run::multi_chain;
use frcw::recom::{RecomParams, RecomProposal, RecomVariant};
use frcw::stats::{SelfLoopCounts, StatsWriter};
use std::collections::HashSet;
use std::io::Result as IOResult;
use std::iter::FromIterator;

use rstest::rstest;
use test_fixtures::{default_fixture, fixture_with_attributes};

/// Verifies that a set of nodes is connected.
fn nodes_connected(graph: &Graph, nodes: &Vec<usize>) -> bool {
    // TODO: This can probably be optimized to use vectors (at the cost of readability?)
    if nodes.is_empty() {
        return true; // ...vacuously.
    }
    // Perform a DFS through the subgraph and attempt to reach all nodes.
    let nodeset = HashSet::<usize>::from_iter(nodes.iter().cloned());
    let mut stack = vec![nodes[0]];
    let mut visited = HashSet::<usize>::from_iter(stack.iter().cloned());
    while let Some(next) = stack.pop() {
        for neighbor in graph.neighbors[next].iter() {
            if nodeset.contains(neighbor) && !visited.contains(neighbor) {
                visited.insert(*neighbor);
                stack.push(*neighbor);
            }
        }
    }
    return visited.len() == nodes.len();
}

/// Verifies that every district in a partition is connected.
fn partition_connected_invariant(graph: &Graph, partition: &Partition) -> bool {
    return partition
        .dist_nodes
        .iter()
        .all(|nodes| nodes_connected(graph, nodes));
}

/// Verifies that the two changed districts in a `RecomProposal` are connected.
fn proposal_connected_invariant(graph: &Graph, proposal: &RecomProposal) -> bool {
    return nodes_connected(graph, &proposal.a_nodes) && nodes_connected(graph, &proposal.b_nodes);
}

/// Verifies all districts in a partition are within population bounds.
fn population_tolerance_invariant(partition: &Partition, min_pop: u32, max_pop: u32) -> bool {
    return partition
        .dist_pops
        .iter()
        .all(|&pop| min_pop <= pop && pop <= max_pop);
}

/// Verifies all districts in a partition have the correct population.
fn population_sum_invariant(graph: &Graph, partition: &Partition) -> bool {
    return partition
        .dist_pops
        .iter()
        .zip(partition.dist_nodes.iter())
        .all(|(&pop, nodes)| pop == nodes.iter().map(|&n| graph.pops[n]).sum::<u32>());
}

/// Verifies that a partition's `cut_edges` match its `assignments`.
fn cut_edges_invariant(graph: &Graph, partition: &mut Partition) -> bool {
    let cut_edges: Vec<usize> = graph
        .edges
        .iter()
        .enumerate()
        .filter(|(_, edge)| partition.assignments[edge.0] != partition.assignments[edge.1])
        .map(|(idx, _)| idx)
        .collect();
    // Technically, it should be OK for the cut edge indices to be in
    // _any_ order. However, it's efficient and simple just to compare
    // vectors (instead of converting to a HashSet) so we might as well
    // do it this way and adjust if necessary.
    return cut_edges == *partition.cut_edges(graph);
}

/// Verifies that a partition's `district_nodes` match its `assignments`.
fn assignments_invariant(graph: &Graph, partition: &Partition) -> bool {
    let node_count: usize = partition.dist_nodes.iter().map(|nodes| nodes.len()).sum();
    if node_count != graph.neighbors.len() {
        return false;
    }
    return partition
        .dist_nodes
        .iter()
        .enumerate()
        .all(|(dist, nodes)| {
            nodes
                .iter()
                .all(|&n| partition.assignments[n as usize] as usize == dist)
        });
}

/// Verifies that a partition's `dist_adj` is consistent with `cut_edges`.
fn dist_adj_invariant(graph: &Graph, partition: &mut Partition) -> bool {
    let n = partition.num_dists;
    let mut dist_adj = vec![0 as u32; (n * n) as usize];
    let cut_edges = partition.cut_edges(graph).clone();
    for &edge_idx in cut_edges.iter() {
        let edge = &graph.edges[edge_idx as usize];
        let dist_a = partition.assignments[edge.0 as usize] as usize;
        let dist_b = partition.assignments[edge.1 as usize] as usize;
        dist_adj[(dist_a * n as usize) + dist_b] += 1;
        dist_adj[(dist_b * n as usize) + dist_a] += 1;
    }
    return dist_adj == *partition.dist_adj(graph);
}

/// Verifies that the global step count was updated properly from the step's counts.
fn step_count_invariant(step: u64, last_step: u64, counts: &SelfLoopCounts) -> bool {
    return step == last_step + counts.sum() as u64 + 1;
}

/// Verifies that a partition's overall properties (number of nodes,
/// total population) are consistent with its graph.
fn graph_partition_invariant(graph: &Graph, partition: &Partition) -> bool {
    return graph.neighbors.len() == partition.assignments.len()
        && graph.total_pop == partition.dist_pops.iter().sum::<u32>();
}

/// The state of a chain, generated from step deltas.
struct StepInvariantWriter {
    /// The chain parameters (relevant: population tolerances).
    params: RecomParams,
    /// The initial partition state.
    /// (`None` if the chain hasn't called .init() yet.)
    initial_partition: Option<Partition>,
    /// The running partition state.
    /// (`None` if the chain hasn't called .init() yet.)
    partition: Option<Partition>,
    /// The global step counter at the last step.
    last_step: u64,
    /// Flag for checking if all districts have changed on close
    /// (that is, no districts are frozen).
    check_frozen: bool
}

impl StepInvariantWriter {
    fn new(params: RecomParams, check_frozen: bool) -> StepInvariantWriter {
        return StepInvariantWriter {
            params: params,
            initial_partition: None,
            partition: None,
            last_step: 0,
            check_frozen: check_frozen
        };
    }
}

/// We observe the state of the chain at each step (and at the initial step)
/// by using `StatsWriter` callbacks (normally used to print chain data to stdout).
impl StatsWriter for StepInvariantWriter {
    /// Checks initial partition invariants and initializes the writer.
    fn init(&mut self, graph: &Graph, partition: &Partition) -> IOResult<()> {
        let mut partition = partition.clone();
        assert!(
            self.initial_partition.is_none(),
            "Writer must be initialized exactly once."
        );
        assert!(
            cut_edges_invariant(graph, &mut partition),
            "Cut edges don't match node assignments in initial partition."
        );
        assert!(
            dist_adj_invariant(graph, &mut partition),
            "Initial partition has incorrect adjacency matrix."
        );
        assert!(
            partition_connected_invariant(graph, &partition),
            "Initial partition is disconnected."
        );
        assert!(
            population_tolerance_invariant(&partition, self.params.min_pop, self.params.max_pop),
            "Initial partition outside population tolerances."
        );
        assert!(
            population_sum_invariant(graph, &partition),
            "District population sums incorrect in initial partition."
        );
        assert!(
            assignments_invariant(graph, &partition),
            ".assignments does not match .dist_nodes in initial partition"
        );
        assert!(
            graph_partition_invariant(graph, &partition),
            "Node count and total population don't match between graph and initial partition."
        );
        self.partition = Some(partition.clone());
        self.initial_partition = Some(partition);
        Ok(())
    }

    /// Checks step-to-step chain invariants (i.e. the validity of each individual proposal).
    fn step(
        &mut self,
        step: u64,
        graph: &Graph,
        partition: &Partition,
        proposal: &RecomProposal,
        counts: &SelfLoopCounts,
    ) -> IOResult<()> {
        assert!(
            self.initial_partition.is_some(),
            "Writer must be initialized exactly once."
        );
        let mut partition = partition.clone();

        // TODO: check that districts in proposal are adjacent.
        assert!(
            cut_edges_invariant(graph, &mut partition),
            "Cut edges don't match node assignment after proposal."
        );
        assert!(
            dist_adj_invariant(graph, &mut partition),
            "District adjacency matrix is incorrect after step."
        );
        assert!(
            proposal_connected_invariant(graph, proposal),
            "At least one of the proposed districts is disconnected."
        );
        assert!(
            population_tolerance_invariant(&partition, self.params.min_pop, self.params.max_pop),
            "Partition outside population tolerances after proposal."
        );
        assert!(
            population_sum_invariant(graph, &partition),
            "District population sums incorrect after proposal."
        );
        assert!(
            graph_partition_invariant(graph, &partition),
            "Node count and total population inconsistent with graph after step."
        );
        assert!(
            assignments_invariant(graph, &partition),
            ".assignments does not match .dist_nodes after proposal."
        );
        assert!(
            step_count_invariant(step, self.last_step, counts),
            "Step count is incorrect after proposal."
        );
        self.partition = Some(partition);
        self.last_step = step;
        Ok(())
    }

    /// Checks start-to-end chain invariants (did things change enough?)
    /// This assumes a large number of steps on a large graph, such that the
    /// probability of any district remaining the same is negligible.
    fn close(&mut self) -> IOResult<()> {
        if self.check_frozen {
            // Check that every district changed (relabeling counts as a change).
            let initial_partition = match self.initial_partition.as_ref() {
                Some(p) => p,
                None => panic!("Writer must be initialized before closing."),
            };
            let last_partition = match self.partition.as_ref() {
                Some(p) => p,
                None => panic!("Writer must be initialized before closing."),
            };
            // TODO: we assume nodes are _literally_ frozen (no permutation)--should we loosen?
            let all_districts_diff = initial_partition
                .dist_nodes
                .iter()
                .zip(last_partition.dist_nodes.iter())
                .all(|(init_nodes, last_nodes)| init_nodes != last_nodes);
            assert!(all_districts_diff, "At least one district frozen!");
        }

        // Enforce closing (sorta).
        self.partition = None;
        self.initial_partition = None;
        Ok(())
    }
}

/// RNG seed for all tests. (TODO: parameterize?)
const RNG_SEED: u64 = 153434375;

#[rstest]
fn test_chain_invariants_recom_grid(
    #[values(2500)] num_steps: u64,
    #[values((6, 6), (5, 7), (4, 8))] pop_range: (u32, u32),
    #[values(RecomVariant::DistrictPairsRMST, RecomVariant::CutEdgesRMST)] variant: RecomVariant,
    #[values(1, 4)] n_threads: usize,
    #[values(1, 4)] batch_size: usize,
) {
    let (graph, partition) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    let params = RecomParams {
        min_pop: pop_range.0,
        max_pop: pop_range.1,
        num_steps: num_steps,
        rng_seed: RNG_SEED,
        balance_ub: 0,
        variant: variant,
        region_weights: None,
    };
    let writer = Box::new(StepInvariantWriter::new(params.clone(), true)) as Box<dyn StatsWriter>;
    multi_chain(&graph, &partition, writer, &params, n_threads, batch_size);
}

#[rstest]
fn test_chain_invariants_revrecom_grid(
    #[values(25000)] num_steps: u64,
    #[values((6, 6), (5, 7), (4, 8))] pop_range: (u32, u32),
    #[values(1, 4)] n_threads: usize,
    #[values(1, 4)] batch_size: usize,
) {
    let (graph, partition) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    let params = RecomParams {
        min_pop: pop_range.0,
        max_pop: pop_range.1,
        num_steps: num_steps,
        rng_seed: RNG_SEED,
        balance_ub: pop_range.1 - pop_range.0 + 1,
        variant: RecomVariant::Reversible,
        region_weights: None,
    };
    let writer = Box::new(StepInvariantWriter::new(params.clone(), true)) as Box<dyn StatsWriter>;
    multi_chain(&graph, &partition, writer, &params, n_threads, batch_size);
}

#[rstest]
fn test_chain_invariants_recom_iowa(
    #[values(0.01, 0.2)] pop_tol: f64,
    #[values(RecomVariant::DistrictPairsRMST, RecomVariant::CutEdgesRMST)] variant: RecomVariant,
    #[values(1, 4)] n_threads: usize,
    #[values(1, 4)] batch_size: usize,
) {
    let (graph, partition) = default_fixture("IA");
    let avg_pop = (graph.total_pop as f64) / (partition.num_dists as f64);
    let params = RecomParams {
        min_pop: ((1.0 - pop_tol) * avg_pop as f64).floor() as u32,
        max_pop: ((1.0 + pop_tol) * avg_pop as f64).ceil() as u32,
        num_steps: 1000,
        rng_seed: RNG_SEED,
        balance_ub: 0,
        variant: variant,
        region_weights: None,
    };
    let writer = Box::new(StepInvariantWriter::new(params.clone(), true)) as Box<dyn StatsWriter>;
    multi_chain(&graph, &partition, writer, &params, n_threads, batch_size);
}

#[rstest]
fn test_chain_invariants_revrecom_iowa(
    #[values((0.01, 10), (0.2, 20))] pop_tol_balance_ub: (f64, u32),
    #[values(1, 4)] n_threads: usize,
    #[values(1, 16)] batch_size: usize,
) {
    let (graph, partition) = default_fixture("IA");
    let avg_pop = (graph.total_pop as f64) / (partition.num_dists as f64);
    let pop_tol = pop_tol_balance_ub.0;
    let balance_ub = pop_tol_balance_ub.1;
    let params = RecomParams {
        min_pop: ((1.0 - pop_tol) * avg_pop as f64).floor() as u32,
        max_pop: ((1.0 + pop_tol) * avg_pop as f64).ceil() as u32,
        num_steps: 20000,
        rng_seed: RNG_SEED,
        balance_ub: balance_ub,
        variant: RecomVariant::Reversible,
        region_weights: None,
    };
    let writer = Box::new(StepInvariantWriter::new(params.clone(), true)) as Box<dyn StatsWriter>;
    multi_chain(&graph, &partition, writer, &params, n_threads, batch_size);
}

#[rstest]
#[ignore]
fn test_chain_invariants_revrecom_large_states(
    #[values("PA", "VA")] state: &str,
    #[values(25000)] num_steps: u64,
    #[values(1, 4)] n_threads: usize,
    #[values(1, 4)] batch_size: usize,
) {
    let (graph, partition) = default_fixture(state);
    let avg_pop = (graph.total_pop as f64) / (partition.num_dists as f64);
    let pop_tol = 0.05;
    let params = RecomParams {
        min_pop: ((1.0 - pop_tol) * avg_pop as f64).floor() as u32,
        max_pop: ((1.0 + pop_tol) * avg_pop as f64).ceil() as u32,
        num_steps: num_steps,
        rng_seed: RNG_SEED,
        balance_ub: 30,
        variant: RecomVariant::Reversible,
        region_weights: None
    };
    let writer = Box::new(StepInvariantWriter::new(params.clone(), false)) as Box<dyn StatsWriter>;
    multi_chain(&graph, &partition, writer, &params, n_threads, batch_size);
}