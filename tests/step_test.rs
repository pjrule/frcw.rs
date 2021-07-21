/// Functional tests that verify ReCom chain invariants at each step.
use std::iter::FromIterator;
use std::collections::{HashMap, HashSet};
use frcw::graph::Graph;
use frcw::partition::Partition;
use frcw::recom::{RecomVariant, RecomParams, RecomProposal};
use frcw::recom::run::multi_chain;
use frcw::stats::{ChainCounts, StatsWriter};

mod fixtures;
use fixtures::fixture_with_attributes;
use rstest::rstest;

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
        .all(|nodes| nodes_connected(graph, nodes))
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
        .all(|&pop| min_pop <= pop && pop <= max_pop)
}

/// Verifies all districts in a partition have the correct population.
fn population_sum_invariant(graph: &Graph, partition: &Partition) -> bool {
    return partition
        .dist_pops
        .iter()
        .zip(partition.dist_nodes.iter())
        .all(|(&pop, nodes)| {
            pop == nodes.iter().map(|&n| graph.pops[n]).sum::<u32>()
        })
}

/// Verifies that a partition's `cut_edges` match its `assignments`.
fn cut_edges_invariant(graph: &Graph, partition: &Partition) -> bool {
    let cut_edges: Vec<usize> = graph
        .edges
        .iter()
        .enumerate()
        .filter(|(_, edge)| partition.assignments[edge.0] != partition.assignments[edge.1])
        .map(|(idx, _)| idx).collect();
    // Technically, it should be OK for the cut edge indices to be in
    // _any_ order. However, it's efficient and simple just to compare
    // vectors (instead of converting to a HashSet) so we might as well
    // do it this way and adjust if necessary.
    return cut_edges == partition.cut_edges;
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
        .all(|(dist, nodes)| nodes.iter().all(|&n| partition.assignments[n as usize] as usize == dist));
}

/// Verifies that a partition's district adjacency matrix is correct.
// TODO

/// Verifies that the global step count was updated properly from the step's counts.

/// Verifies that a partition's overall properties (e.g. number of nodes)
/// are consistent with its graph.

/// The state of a chain, generated from step deltas.
struct StepInvariantWriter {
    /// The chain parameters (relevant: population tolerances).
    params: RecomParams,
    /// The global sums of attributes.
    attr_sums: HashMap<String, u32>,
    /// The running partition state.
    /// (`None` if the chain hasn't called .init() yet.)
    partition: Option<Partition>,
    /// The global step counter at the last step.
    last_step: u64,
}

impl StepInvariantWriter {
    fn new(params: RecomParams) -> StepInvariantWriter {
        return StepInvariantWriter {
            params: params,
            attr_sums: HashMap::new(),
            partition: None,
            last_step: 0,
        };
    }
}

/// We observe the state of the chain at each step (and at the initial step)
/// by using `StatsWriter` callbacks (normally used to print chain data to stdout).
impl StatsWriter for StepInvariantWriter {
    /// Checks initial partition invariants and initializes the writer.
    fn init(&mut self, graph: &Graph, partition: &Partition) {
        assert!(
            self.partition.is_none(),
            "Writer must be initialized exactly once."
        );
        self.attr_sums = graph
            .attr
            .iter()
            .map(|(key, values)| (key.clone(), values.iter().sum()))
            .collect();
        self.partition = Some(partition.clone());
        assert!(
            partition_connected_invariant(graph, partition),
            "Initial partition is disconnected."
        );
        assert!(
            population_tolerance_invariant(
                partition,
                self.params.min_pop,
                self.params.max_pop
            ),
            "Initial partition outside population tolerances."
        );
        // TODO: more here.
    }
        
    fn step(&mut self, step: u64, graph: &Graph, proposal: &RecomProposal, counts: &ChainCounts) {
        assert!(
            self.partition.is_some(),
            "Writer must be initialized before writing a step delta."
        );
        let Some(mut partition) = self.partition.as_mut();

        // TODO: check that districts in proposal are adjacent.
        assert!(
            proposal_connected_invariant(graph, proposal), 
            "At least one of the proposed districts is disconnected."
        );
        // TODO: more here.
        partition.update(graph, proposal);
        self.last_step = step;
    }
}

/// RNG seed for all tests. (TODO: parameterize?)
const RNG_SEED: u64 = 153434375;

#[rstest]
fn test_chain_invariants_recom_grid(
    #[values(10000)]
    num_steps: u64,
    #[values((6, 6), (5, 7), (4, 8))]
    pop_range: (u32, u32),
    #[values(RecomVariant::DistrictPairs, RecomVariant::CutEdges)]
    variant: RecomVariant,
    #[values(1, 4)]
    n_threads: usize,
    #[values(1, 4)]
    batch_size: usize
) {
    let params = RecomParams {
        min_pop: pop_range.0,
        max_pop: pop_range.1,
        num_steps: num_steps,
        rng_seed: RNG_SEED,
        balance_ub: 0,
        variant: variant
    };
    let (graph, partition) = fixture_with_attributes(
        "6x6", vec!["a_share", "b_share"]
    );
    let writer = Box::new(StepInvariantWriter::new(params)) as Box<dyn StatsWriter>;
    multi_chain(&graph, &partition, writer, params, n_threads, batch_size);
}
