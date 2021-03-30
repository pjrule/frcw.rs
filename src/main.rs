extern crate jemallocator;
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

extern crate clap;
extern crate rand;
extern crate serde_json;
use clap::{value_t, App, Arg};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde_json::Result as SerdeResult;
use serde_json::Value;
use std::cmp::{max, min};
use std::collections::VecDeque;
use std::fs;
use std::result::Result;
use std::boxed::Box;
use std::num::Wrapping;


type MST = Vec<Vec<usize>>;

#[derive(Clone, Hash, Eq, PartialEq)]
struct Edge(usize, usize);

struct Node {
    pop: u32,
    neighbors: Vec<usize>,
}

struct Graph {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    edges_start: Vec<usize>,
    total_pop: u32,
}

struct Partition {
    num_dists: u32,
    assignments: Vec<u32>,
    cut_edges: Vec<usize>,
    dist_adj: Vec<u32>,
    dist_pops: Vec<u32>,
    dist_nodes: Vec<Vec<usize>>,
}

struct RecomProposal {
    a_label: usize,
    b_label: usize,
    a_pop: u32,
    b_pop: u32,
    a_nodes: Vec<usize>,
    b_nodes: Vec<usize>,
}

#[derive(Copy, Clone)]
struct ChainParams {
    min_pop: u32,
    max_pop: u32,
    M: u32,
    num_steps: u64,
    rng_seed: u64,
}

struct ChainState {
    non_adjacent: u32,
    no_split: u32,
    seam_length: u32,
}

impl Default for ChainState {
    fn default() -> ChainState {
        ChainState {
            non_adjacent: 0,
            no_split: 0,
            seam_length: 0,
        }
    }
}


fn from_networkx(
    path: &str,
    pop_col: &str,
    assignment_col: &str,
) -> SerdeResult<(Graph, Partition)> {
    let raw = fs::read_to_string(path).expect("Could not load graph");
    let data: Value = serde_json::from_str(&raw)?;

    let raw_nodes = data["nodes"].as_array().unwrap();
    let raw_adj = data["adjacency"].as_array().unwrap();
    let num_nodes = raw_nodes.len();
    let mut nodes = Vec::<Node>::with_capacity(num_nodes);
    let mut assignments = Vec::<u32>::with_capacity(num_nodes);
    let mut edges = Vec::<Edge>::new();
    let mut edges_start = vec![0 as usize; num_nodes];

    for (index, (node, adj)) in raw_nodes.iter().zip(raw_adj.iter()).enumerate() {
        edges_start[index] = edges.len();
        let neighbors: Vec<usize> = adj
            .as_array()
            .unwrap()
            .into_iter()
            .map(|n| n.as_object().unwrap()["id"].as_u64().unwrap() as usize)
            .collect();
        // TODO: validate population
        nodes.push(Node {
            pop: node[pop_col].as_u64().unwrap() as u32,
            neighbors: neighbors.clone(),
        });
        assignments.push((node[assignment_col].as_u64().unwrap() - 1) as u32); // TODO: 1-indexing vs. 0-indexing
        for neighbor in &neighbors {
            if neighbor > &index {
                let edge = Edge(index, *neighbor);
                edges.push(edge.clone());
            }
        }
    }

    let total_pop = nodes.iter().map(|n| n.pop).sum();
    let num_dists = assignments.iter().max().unwrap() + 1;
    let mut dist_nodes: Vec<Vec<usize>> = (0..num_dists).map(|_| Vec::<usize>::new()).collect();
    for (index, assignment) in assignments.iter().enumerate() {
        assert!(assignment < &num_dists);
        dist_nodes[*assignment as usize].push(index);
    }
    let mut dist_adj = vec![0 as u32; (num_dists * num_dists) as usize];
    let mut cut_edges = Vec::<usize>::new();
    for (index, edge) in edges.iter().enumerate() {
        let dist_a = assignments[edge.0 as usize];
        let dist_b = assignments[edge.1 as usize];
        assert!(dist_a < num_dists);
        assert!(dist_b < num_dists);
        if dist_a != dist_b {
            dist_adj[((dist_a * num_dists) + dist_b) as usize] += 1;
            dist_adj[((dist_b * num_dists) + dist_a) as usize] += 1;
            cut_edges.push(index);
        }
    }
    let mut dist_pops = vec![0 as u32; num_dists as usize];
    for (index, node) in nodes.iter().enumerate() {
        dist_pops[assignments[index] as usize] += node.pop;
    }

    let graph = Graph {
        nodes: nodes,
        edges: edges.clone(),
        edges_start: edges_start.clone(),
        total_pop: total_pop,
    };
    let partition = Partition {
        num_dists: num_dists,
        assignments: assignments,
        cut_edges: cut_edges,
        dist_adj: dist_adj,
        dist_pops: dist_pops,
        dist_nodes: dist_nodes,
    };
    return Ok((graph, partition));
}

impl RecomProposal {
    pub fn seam_length(&self, graph: &Graph) -> usize {
        let mut a_mask = vec![false; graph.nodes.len()];
        for &node in self.a_nodes.iter() {
            a_mask[node] = true;
        }
        let mut seam = 0;
        for &node in self.b_nodes.iter() {
            for &neighbor in graph.nodes[node].neighbors.iter() {
                if a_mask[neighbor] {
                    seam += 1;
                }
            }
        }
        return seam;
    }
}

impl Partition {
    pub fn update(&mut self, graph: &Graph, proposal: &RecomProposal) {
        // Move nodes.
        self.dist_nodes[proposal.a_label] = proposal.a_nodes.clone();
        self.dist_nodes[proposal.b_label] = proposal.b_nodes.clone();
        self.dist_pops[proposal.a_label] = proposal.a_pop;
        self.dist_pops[proposal.b_label] = proposal.b_pop;
        for &node in proposal.a_nodes.iter() {
            self.assignments[node] = proposal.a_label as u32;
        }
        for &node in proposal.b_nodes.iter() {
            self.assignments[node] = proposal.b_label as u32;
        }
        // Recompute adjacency/cut edges.
        let mut dist_adj = vec![0 as u32; (self.num_dists * self.num_dists) as usize];
        let mut cut_edges = Vec::<usize>::new();
        for (index, edge) in graph.edges.iter().enumerate() {
            let dist_a = self.assignments[edge.0 as usize];
            let dist_b = self.assignments[edge.1 as usize];
            assert!(dist_a < self.num_dists);
            assert!(dist_b < self.num_dists);
            if dist_a != dist_b {
                dist_adj[((dist_a * self.num_dists) + dist_b) as usize] += 1;
                dist_adj[((dist_b * self.num_dists) + dist_a) as usize] += 1;
                cut_edges.push(index);
            }
        }
        self.dist_adj = dist_adj;
        self.cut_edges = cut_edges;
    }
    pub fn subgraph(&self, graph: &Graph, a: usize, b: usize) -> (Box<Graph>, Box<Vec<usize>>) {
        let raw_nodes: Vec<usize> = self.dist_nodes[a]
            .iter()
            .cloned()
            .chain(self.dist_nodes[b].iter().cloned())
            .collect();
        let n = raw_nodes.len();
        let mut node_to_idx = vec![-1 as i64; graph.nodes.len()];
        let mut nodes = Vec::<Node>::with_capacity(n);
        let mut edges = Vec::<Edge>::with_capacity(3 * n);
        let mut edges_start = vec![0 as usize; n];
        for (idx, &node) in raw_nodes.iter().enumerate() {
            node_to_idx[node] = idx as i64;
        }
        for (idx, &node) in raw_nodes.iter().enumerate() {
            edges_start[idx] = edges.len();
            let mut neighbors = Vec::<usize>::new();
            for &neighbor in graph.nodes[node].neighbors.iter() {
                if node_to_idx[neighbor] >= 0 {
                    let neighbor_idx = node_to_idx[neighbor] as usize;
                    neighbors.push(neighbor_idx as usize);
                    if neighbor_idx > idx as usize {
                        edges.push(Edge(idx, neighbor_idx as usize));
                    }
                }
            }
            nodes.push(Node { pop: graph.nodes[node].pop, neighbors: neighbors });
        }
        let graph = Graph {
            nodes: nodes,
            edges: edges,
            edges_start: edges_start,
            total_pop: self.dist_pops[a] + self.dist_pops[b]
        };
        return (Box::new(graph), Box::new(raw_nodes));
    }
    pub fn invariants(&self) -> bool {
        return self.contiguous() && self.pops_in_tolerance() && self.consec_labels();
    }
    fn contiguous(&self) -> bool {
        // TODO: invariant check (optional).
        false
    }
    fn pops_in_tolerance(&self) -> bool {
        // TODO: invariant check (optional).
        false
    }
    fn consec_labels(&self) -> bool {
        // TODO: invariant check (optional).
        false
    }
}

fn rand_in_range(rng: &mut SmallRng, ub: u32) -> u32 {
    // https://www.pcg-random.org/posts/bounded-rands.html
    let mut t = (Wrapping(0) - Wrapping(ub)).0 % ub;
    let mut x = rng.gen::<u32>();
    let mut m = (x as u64) * (ub as u64);
    let mut l = m as u32;
    while l < t {
        x = rng.gen::<u32>();
        m = (x as u64) * (ub as u64);
        l = Wrapping(m).0 as u32;
    }
    return Wrapping(m >> 32).0 as u32;
}

fn random_spanning_tree(graph: &Graph, rng: &mut SmallRng) -> Box<MST> {
    let n = graph.nodes.len();
    let mut in_tree = vec![false; n];
    let mut next = vec![-1 as i64; n];
    let root = rng.gen_range(0..n);
    in_tree[root] = true;
    for i in 0..n {
        let mut u = i;
        while !in_tree[u] {
            let neighbors = &graph.nodes[u].neighbors;
            let neighbor = neighbors[rand_in_range(rng, neighbors.len() as u32) as usize];
            next[u] = neighbor as i64;
            u = neighbor;
        }
        u = i;
        while !in_tree[u] {
            in_tree[u] = true;
            u = next[u] as usize;
        }
    }

    let mut mst_edges = Vec::<usize>::with_capacity(n - 1);
    for (curr, &prev) in next.iter().enumerate() {
        if prev >= 0 {
            let a = min(curr, prev as usize);
            let b = max(curr, prev as usize);
            let mut edge_idx = graph.edges_start[a];
            while graph.edges[edge_idx].0 == a {
                if graph.edges[edge_idx].1 == b {
                    mst_edges.push(edge_idx);
                    break;
                }
                edge_idx += 1;
            }
        }
    }
    assert!(mst_edges.len() == n - 1);

    let mut mst = vec![Vec::<usize>::new(); n];
    for &edge in mst_edges.iter() {
        let Edge(src, dst) = graph.edges[edge];
        mst[src].push(dst);
        mst[dst].push(src);
    }
    return Box::new(mst);
}

fn random_split(
    subgraph: &Graph,
    partition: &Partition,
    rng: &mut SmallRng,
    mst: &MST,
    a: usize,
    b: usize,
    subgraph_map: &Vec<usize>,
    params: ChainParams,
) -> Result<Box<RecomProposal>, String> {
    let n = subgraph.nodes.len();
    let mut root = 0;
    while root < n {
        if subgraph.nodes[root].neighbors.len() > 1 {
            break;
        }
        root += 1;
    }
    if root == n {
        return Err("no leaf nodes in MST".to_string());
    }
    // Traverse the MST.
    let mut visited = vec![false; n];
    let mut pred = vec![0; n];
    let mut succ = vec![Vec::<usize>::new(); n];
    let mut queue = VecDeque::<usize>::with_capacity(n);
    queue.push_back(root);
    while let Some(next) = queue.pop_front() {
        visited[next] = true;
        for &neighbor in mst[next].iter() {
            if !visited[neighbor] {
                queue.push_back(neighbor);
                succ[next].push(neighbor);
                pred[neighbor] = next;
            }
        }
    }

    // Recursively compute populations of subtrees.
    let mut tree_pops = vec![0 as u32; n];
    let mut pop_found = vec![false; n];
    let mut stack = Vec::<usize>::with_capacity(n);
    stack.push(root);
    while let Some(next) = stack.pop() {
        if !pop_found[next] {
            if subgraph.nodes[next].neighbors.len() == 1 {
                tree_pops[next] = subgraph.nodes[next].pop;
                pop_found[next] = true;
            } else {
                // Populations of all child nodes found. :)
                if succ[next].iter().all(|&node| pop_found[node]) {
                    tree_pops[next] = succ[next].iter().map(|&node| tree_pops[node]).sum();
                    tree_pops[next] += subgraph.nodes[next].pop;
                    pop_found[next] = true;
                } else {
                    // Come back later.
                    stack.push(next);
                    for &neighbor in succ[next].iter() {
                        if !pop_found[neighbor] {
                            stack.push(neighbor);
                        }
                    }
                }
            }
        }
    }

    // Find ε-balanced cuts.
    let mut balance_nodes = Vec::<usize>::new();
    for (index, &pop) in tree_pops.iter().enumerate() {
        if pop >= params.min_pop
            && pop <= params.max_pop
            && subgraph.total_pop - pop >= params.min_pop
            && subgraph.total_pop - pop <= params.max_pop
        {
            balance_nodes.push(index);
        }
    }
    if balance_nodes.is_empty() {
        return Err("no balanced cuts".to_string());
    } else if balance_nodes.len() > params.M as usize {
        panic!(
            "Reversibility invariant violated: expected ≤{} balanced cuts, found {}",
            params.M,
            balance_nodes.len()
        );
    }
    let balance_node = balance_nodes[rng.gen_range(0..balance_nodes.len())];
    queue.push_back(balance_node);

    // Extract the nodes for a random cut.
    let mut a_nodes = Vec::<usize>::with_capacity(n);
    let mut a_pop = 0;
    let mut in_a = vec![false; n];
    while let Some(next) = queue.pop_front() {
        if !in_a[next] {
            a_nodes.push(next);
            a_pop += subgraph.nodes[next].pop;
            in_a[next] = true;
            for &node in succ[next].iter() {
                queue.push_back(node);
            }
        }
    }
    let mut b_nodes = Vec::<usize>::with_capacity(n - a_nodes.len());
    for (index, &e) in in_a.iter().enumerate() {
        if !e {
            b_nodes.push(index);
        }
    }
    return Ok(Box::new(RecomProposal {
        a_label: a,
        b_label: b,
        a_nodes: a_nodes.iter().map(|&n| subgraph_map[n]).collect(),
        b_nodes: b_nodes.iter().map(|&n| subgraph_map[n]).collect(),
        a_pop: a_pop,
        b_pop: subgraph.total_pop - a_pop,
    }));
}

fn run_chain(graph: &Graph, partition: &mut Partition, params: ChainParams) {
    let mut step = 0;
    let mut state = ChainState::default();
    let mut rng: SmallRng = SeedableRng::seed_from_u64(params.rng_seed);
    while step <= params.num_steps {
        step += 1;
        //println!("step {}", step);
        // Step 1: randomly sample from the n^2 district pairs.
        let dist_a = rng.gen_range(0..partition.num_dists) as usize;
        let dist_b = rng.gen_range(0..partition.num_dists) as usize;
        if partition.dist_adj[(dist_a * partition.num_dists as usize) + dist_b] == 0 {
            state.non_adjacent += 1; // Self-loop.
            continue;
        }
        let (subgraph, subgraph_map) = partition.subgraph(graph, dist_a, dist_b);
        // Step 2: draw a random spanning tree of the subgraph induced by the
        // two districts.
        let mst = random_spanning_tree(&subgraph, &mut rng);
        // Step 3: choose a random balance edge, if possible.
        let split = random_split(&subgraph, partition, &mut rng, &mst, dist_a, dist_b, &subgraph_map, params);
        match split {
            Ok(proposal) => {
                // Step 4: accept with probability 1 / (M * seam length)
                let seam_length = proposal.seam_length(graph);
                if rng.gen::<f64>() < 1.0 / (seam_length as f64 * params.M as f64) { 
                    partition.update(graph, &proposal);
                    println!("accepted!");
                    state = ChainState::default();
                } else {
                    state.seam_length += 1;
                }
            },
            Err(_) => state.no_split += 1  // TODO: break out errors?
        }
    }
}

fn main() {
    let matches = App::new("frcw")
        .version("0.1.0")
        .author("Parker J. Rule <parker.rule@tufts.edu>")
        .about("A minimal implementation of the reversible ReCom Markov chain")
        .arg(
            Arg::with_name("graph_json")
                .long("graph-json")
                .takes_value(true)
                .required(true)
                .help("The path of the dual graph (in NetworkX format)."),
        )
        .arg(
            Arg::with_name("n_steps")
                .long("n-steps")
                .takes_value(true)
                .required(true)
                .help("The number of proposals to generate."),
        )
        .arg(
            Arg::with_name("tol")
                .long("tol")
                .takes_value(true)
                .required(true)
                .help("The relative population tolerance."),
        )
        .arg(
            Arg::with_name("pop_col")
                .long("pop-col")
                .takes_value(true)
                .required(true)
                .help("The name of the total population column in the graph metadata."),
        )
        .arg(
            Arg::with_name("assignment_col")
                .long("assignment-col")
                .takes_value(true)
                .required(true)
                .help("The name of the assignment column in the graph metadata."),
        )
        .arg(
            Arg::with_name("rng_seed")
                .long("rng-seed")
                .takes_value(true)
                .required(true)
                .help("The seed of the RNG used to draw proposals."),
        )
        .arg(
            Arg::with_name("M")
                .long("M")
                .takes_value(true)
                .required(true)
                .help("The normalizing constant for reversibility."),
        )
        .get_matches();
    let n_steps = value_t!(matches.value_of("n_steps"), u64).unwrap_or_else(|e| e.exit());
    let rng_seed = value_t!(matches.value_of("rng_seed"), u64).unwrap_or_else(|e| e.exit());
    let tol = value_t!(matches.value_of("tol"), f64).unwrap_or_else(|e| e.exit());
    let M = value_t!(matches.value_of("M"), u32).unwrap_or_else(|e| e.exit());
    assert!(tol >= 0.0 && tol <= 1.0);

    let (graph, mut partition) = from_networkx(
        matches.value_of("graph_json").unwrap(),
        matches.value_of("pop_col").unwrap(),
        matches.value_of("assignment_col").unwrap(),
    )
    .unwrap();
    let avg_pop = (graph.total_pop as f64) / (partition.num_dists as f64);
    let params = ChainParams {
        min_pop: ((1.0 - tol) * avg_pop as f64).floor() as u32,
        max_pop: ((1.0 + tol) * avg_pop as f64).ceil() as u32,
        num_steps: n_steps,
        rng_seed: rng_seed,
        M: M,
    };
    run_chain(&graph, &mut partition, params);
}
