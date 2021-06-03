extern crate jemallocator;
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

extern crate clap;
extern crate rand;
extern crate serde_json;
extern crate crossbeam;
extern crate crossbeam_channel;
use clap::{value_t, App, Arg};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde_json::Result as SerdeResult;
use serde_json::{json, Value};
use std::cmp::{max, min};
use std::collections::VecDeque;
use std::{fs, io};
use std::path::PathBuf;
use std::num::Wrapping;
use std::result::Result;
use std::ops::Add;
use crossbeam_channel::unbounded;
use sha3::{Sha3_256, Digest};

type MST = Vec<Vec<usize>>;

#[derive(Clone, Hash, Eq, PartialEq)]
struct Edge(usize, usize);

#[derive(Clone)]
struct Graph {
    edges: Vec<Edge>,
    pops: Vec<u32>,
    neighbors: Vec<Vec<usize>>,
    edges_start: Vec<usize>,
    total_pop: u32,
}

#[derive(Clone)]
struct Partition {
    num_dists: u32,
    assignments: Vec<u32>,
    cut_edges: Vec<usize>,
    dist_adj: Vec<u32>,
    dist_pops: Vec<u32>,
    dist_nodes: Vec<Vec<usize>>,
}

#[derive(Clone)]
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

#[derive(Copy, Clone, Default)]
struct ChainCounts {
    non_adjacent: usize,
    no_split: usize,
    seam_length: usize
}

impl Add for ChainCounts {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            non_adjacent: self.non_adjacent + other.non_adjacent,
            no_split: self.no_split + other.no_split,
            seam_length: self.seam_length + other.seam_length
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
    let mut pops = Vec::<u32>::with_capacity(num_nodes);
    let mut neighbors = Vec::<Vec<usize>>::with_capacity(num_nodes);
    let mut assignments = Vec::<u32>::with_capacity(num_nodes);
    let mut edges = Vec::<Edge>::new();
    let mut edges_start = vec![0 as usize; num_nodes];

    for (index, (node, adj)) in raw_nodes.iter().zip(raw_adj.iter()).enumerate() {
        edges_start[index] = edges.len();
        let node_neighbors: Vec<usize> = adj
            .as_array()
            .unwrap()
            .into_iter()
            .map(|n| n.as_object().unwrap()["id"].as_u64().unwrap() as usize)
            .collect();
        pops.push(node[pop_col].as_u64().unwrap() as u32);
        neighbors.push(node_neighbors.clone());
        assignments.push((node[assignment_col].as_u64().unwrap() - 1) as u32); // TODO: 1-indexing vs. 0-indexing
        for neighbor in &node_neighbors {
            if neighbor > &index {
                let edge = Edge(index, *neighbor);
                edges.push(edge.clone());
            }
        }
    }

    let total_pop = pops.iter().sum();
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
    for (index, pop) in pops.iter().enumerate() {
        dist_pops[assignments[index] as usize] += pop;
    }

    let graph = Graph {
        pops: pops,
        neighbors: neighbors,
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
    pub fn new_buffer(n: usize) -> RecomProposal {
        return RecomProposal {
            a_label: 0,
            b_label: 0,
            a_pop: 0,
            b_pop: 0,
            a_nodes: Vec::<usize>::with_capacity(n),
            b_nodes: Vec::<usize>::with_capacity(n),
        };
    }

    pub fn clear(&mut self) {
        self.a_nodes.clear();
        self.b_nodes.clear();
        // TODO: reset integer fields?
    }

    pub fn seam_length(&self, graph: &Graph) -> usize {
        let mut a_mask = vec![false; graph.pops.len()];
        for &node in self.a_nodes.iter() {
            a_mask[node] = true;
        }
        let mut seam = 0;
        for &node in self.b_nodes.iter() {
            for &neighbor in graph.neighbors[node].iter() {
                if a_mask[neighbor] {
                    seam += 1;
                }
            }
        }
        return seam;
    }
}

impl Graph {
    pub fn new_buffer(n: usize) -> Graph {
        return Graph {
            pops: Vec::<u32>::with_capacity(n),
            neighbors: vec![Vec::<usize>::with_capacity(8); n],
            edges: Vec::<Edge>::with_capacity(8 * n),
            edges_start: vec![0 as usize; n],
            total_pop: 0,
        };
    }

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

struct SubgraphBuffer {
    raw_nodes: Vec<usize>,
    node_to_idx: Vec<i64>,
    graph: Graph,
}

impl SubgraphBuffer {
    pub fn new(n: usize, b: usize) -> SubgraphBuffer {
        return SubgraphBuffer {
            raw_nodes: Vec::<usize>::with_capacity(b),
            node_to_idx: vec![-1 as i64; n],
            graph: Graph::new_buffer(b),
        };
    }

    pub fn clear(&mut self) {
        self.raw_nodes.clear();
        self.node_to_idx.fill(-1);
        self.graph.clear();
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
    pub fn subgraph(&self, graph: &Graph, buf: &mut SubgraphBuffer, a: usize, b: usize) {
        buf.clear();
        for &node in self.dist_nodes[a].iter() {
            buf.raw_nodes.push(node);
        }
        for &node in self.dist_nodes[b].iter() {
            buf.raw_nodes.push(node);
        }
        for (idx, &node) in buf.raw_nodes.iter().enumerate() {
            buf.node_to_idx[node] = idx as i64;
        }
        let mut edge_pos = 0;
        for (idx, &node) in buf.raw_nodes.iter().enumerate() {
            buf.graph.edges_start[idx] = edge_pos;
            for &neighbor in graph.neighbors[node].iter() {
                if buf.node_to_idx[neighbor] >= 0 {
                    let neighbor_idx = buf.node_to_idx[neighbor] as usize;
                    buf.graph.neighbors[idx].push(neighbor_idx as usize);
                    if neighbor_idx > idx as usize {
                        buf.graph.edges.push(Edge(idx, neighbor_idx as usize));
                        edge_pos += 1;
                    }
                }
            }
            buf.graph.pops.push(graph.pops[node]);
        }
        buf.graph.total_pop = self.dist_pops[a] + self.dist_pops[b];
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

const RANGE_BUF_SIZE: usize = 1 << 20;
struct RandomRangeBuffer {
    buf: Vec<u8>,
    size: usize,
    pos: usize,
}

impl RandomRangeBuffer {
    fn new(rng: &mut SmallRng) -> RandomRangeBuffer {
        let mut buf = vec![0 as u8; RANGE_BUF_SIZE];
        rng.fill(&mut buf[..]);
        return RandomRangeBuffer {
            buf: buf,
            size: RANGE_BUF_SIZE,
            pos: 0,
        }
    }

    fn next(&mut self, rng: &mut SmallRng) -> u8 {
        let val = self.buf[self.pos];
        self.pos += 1;
        if self.pos == self.size {
            rng.fill(&mut self.buf[..]);
            self.pos = 0;
        }
        return val;
    }

    fn range(&mut self, rng: &mut SmallRng, ub: u8) -> u8 {
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

struct MSTBuffer {
    in_tree: Vec<bool>,
    next: Vec<i64>,
    mst_edges: Vec<usize>,
    mst: Vec<Vec<usize>>,
}

impl MSTBuffer {
    pub fn new(n: usize) -> MSTBuffer {
        return MSTBuffer {
            in_tree: vec![false; n],
            next: vec![-1 as i64; n],
            mst_edges: Vec::<usize>::with_capacity(n - 1),
            mst: vec![Vec::<usize>::with_capacity(8); n],
        };
    }

    pub fn clear(&mut self) {
        self.in_tree.fill(false);
        self.next.fill(-1);
        self.mst_edges.clear();
        for node in self.mst.iter_mut() {
            node.clear();
        }
    }
}

fn random_spanning_tree(graph: &Graph, buf: &mut MSTBuffer, range_buf: &mut RandomRangeBuffer, rng: &mut SmallRng) {
    buf.clear();
    let n = graph.pops.len();
    let root = rng.gen_range(0..n);
    buf.in_tree[root] = true;
    for i in 0..n {
        let mut u = i;
        while !buf.in_tree[u] {
            let neighbors = &graph.neighbors[u];
            let neighbor = neighbors[range_buf.range(rng, neighbors.len() as u8) as usize];
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
                    buf.mst_edges.push(edge_idx);
                    break;
                }
                edge_idx += 1;
            }
        }
    }
    if buf.mst_edges.len() != n - 1 {
        panic!("expected to have {} edges in MST but got {}", n - 1, buf.mst_edges.len());
    }
    //assert!(buf.mst_edges.len() == n - 1);

    for &edge in buf.mst_edges.iter() {
        let Edge(src, dst) = graph.edges[edge];
        buf.mst[src].push(dst);
        buf.mst[dst].push(src);
    }
}

struct SplitBuffer {
    visited: Vec<bool>,
    pred: Vec<usize>,
    succ: Vec<Vec<usize>>,
    deque: VecDeque<usize>,
    tree_pops: Vec<u32>,
    pop_found: Vec<bool>,
    balance_nodes: Vec<usize>,
    in_a: Vec<bool>,
}

impl SplitBuffer {
    fn new(n: usize, m: usize) -> SplitBuffer {
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
    fn clear(&mut self) {
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

fn random_split(
    subgraph: &Graph,
    rng: &mut SmallRng,
    mst: &MST,
    a: usize,
    b: usize,
    buf: &mut SplitBuffer,
    proposal: &mut RecomProposal,
    subgraph_map: &Vec<usize>,
    params: &ChainParams,
) -> Result<usize, String> {
    buf.clear();
    proposal.clear();
    let n = subgraph.pops.len();
    let mut root = 0;
    while root < n {
        if subgraph.neighbors[root].len() > 1 {
            break;
        }
        root += 1;
    }
    if root == n {
        return Err("no leaf nodes in MST".to_string());
    }
    // Traverse the MST.
    buf.deque.push_back(root);
    while let Some(next) = buf.deque.pop_front() {
        buf.visited[next] = true;
        for &neighbor in mst[next].iter() {
            if !buf.visited[neighbor] {
                buf.deque.push_back(neighbor);
                buf.succ[next].push(neighbor);
                buf.pred[neighbor] = next;
            }
        }
    }

    // Recursively compute populations of subtrees.
    buf.deque.push_back(root);
    while let Some(next) = buf.deque.pop_back() {
        if !buf.pop_found[next] {
            if subgraph.neighbors[next].len() == 1 {
                buf.tree_pops[next] = subgraph.pops[next];
                buf.pop_found[next] = true;
            } else {
                // Populations of all child nodes found. :)
                if buf.succ[next].iter().all(|&node| buf.pop_found[node]) {
                    buf.tree_pops[next] =
                        buf.succ[next].iter().map(|&node| buf.tree_pops[node]).sum();
                    buf.tree_pops[next] += subgraph.pops[next];
                    buf.pop_found[next] = true;
                } else {
                    // Come back later.
                    buf.deque.push_back(next);
                    for &neighbor in buf.succ[next].iter() {
                        if !buf.pop_found[neighbor] {
                            buf.deque.push_back(neighbor);
                        }
                    }
                }
            }
        }
    }

    // Find ε-balanced cuts.
    let mut prob_correction = 1.0;
    for (index, &pop) in buf.tree_pops.iter().enumerate() {
        if pop >= params.min_pop
            && pop <= params.max_pop
            && subgraph.total_pop - pop >= params.min_pop
            && subgraph.total_pop - pop <= params.max_pop
        {
            buf.balance_nodes.push(index);
        }
    }
    if buf.balance_nodes.is_empty() {
        return Err("no balanced cuts".to_string());
    } else if buf.balance_nodes.len() > params.M as usize {
        println!("Warning: found {} balance nodes (soft upper bound {})",
                 buf.balance_nodes.len(), params.M);
    }
    let balance_node = buf.balance_nodes[rng.gen_range(0..buf.balance_nodes.len())];
    buf.deque.push_back(balance_node);

    // Extract the nodes for a random cut.
    let mut a_pop = 0;
    while let Some(next) = buf.deque.pop_front() {
        if !buf.in_a[next] {
            proposal.a_nodes.push(subgraph_map[next]);
            a_pop += subgraph.pops[next];
            buf.in_a[next] = true;
            for &node in buf.succ[next].iter() {
                buf.deque.push_back(node);
            }
        }
    }
    for index in 0..n {
        if !buf.in_a[index] {
            proposal.b_nodes.push(subgraph_map[index]);
        }
    }
    proposal.a_label = a;
    proposal.b_label = b;
    proposal.a_pop = a_pop;
    proposal.b_pop = subgraph.total_pop - a_pop;
    return Ok((buf.balance_nodes.len()));
}

fn node_bound(pops: &Vec<u32>, max_pop: u32) -> usize {
    let mut sorted_pops = pops.clone();
    sorted_pops.sort();
    let mut node_bound = 0;
    let mut total = 0;
    while total < 2 * max_pop {
        total += sorted_pops[node_bound];
        node_bound += 1;
    }
    return node_bound + 1;
}

struct JobPacket {
    n_steps: usize,
    diff: Option<RecomProposal>,
    terminate: bool
}

struct ResultPacket {
    counts: ChainCounts,
    proposals: Vec<RecomProposal>,
}

fn run_multi_chain(graph: &Graph, partition: &Partition, params: ChainParams, n_threads: usize, batch_size: usize) {
    let mut step = 0;
    let mut state = ChainCounts::default();
    let node_ub = node_bound(&graph.pops, params.max_pop);
    let mut job_sends = vec![];
    let mut job_recvs = vec![];
    for _ in 0..n_threads { // TODO: fancy unzipping?
        let (s, r) = unbounded();
        job_sends.push(s);
        job_recvs.push(r); 
    }
    let (result_send, result_recv) = unbounded();
    let mut rng: SmallRng = SeedableRng::seed_from_u64(params.rng_seed);

    // Start threads.
    crossbeam::scope(|scope| {
        for t_idx in 0..n_threads {
            let job_r = job_recvs[t_idx].clone();
            let res_s = result_send.clone();
            let graph = graph.clone();
            let mut partition = partition.clone();
            scope.spawn(move |_| {
                let n = graph.pops.len();
                let mut rng: SmallRng = SeedableRng::seed_from_u64(params.rng_seed + t_idx as u64 + 1); // TODO: something better
                let mut subgraph_buf = SubgraphBuffer::new(n, node_ub);
                let mut mst_buf = MSTBuffer::new(node_ub);
                let mut split_buf = SplitBuffer::new(node_ub, params.M as usize);
                let mut proposal_buf = RecomProposal::new_buffer(node_ub);
                let mut range_buf = RandomRangeBuffer::new(&mut rng);

                let mut next: JobPacket = job_r.recv().unwrap();
                while !next.terminate {
                    match next.diff {
                        Some(diff) => partition.update(&graph, &diff),
                        None => {}
                    }
                    let mut counts = ChainCounts::default();
                    let mut proposals = Vec::<RecomProposal>::new();
                    for _ in 0..next.n_steps {
                        let dist_a = rng.gen_range(0..partition.num_dists) as usize;
                        let dist_b = rng.gen_range(0..partition.num_dists) as usize;
                        if partition.dist_adj[(dist_a * partition.num_dists as usize) + dist_b] == 0 {
                            counts.non_adjacent += 1; // Self-loop.
                            continue;
                        }
                        partition.subgraph(&graph, &mut subgraph_buf, dist_a, dist_b);
                        // Step 2: draw a random spanning tree of the subgraph induced by the
                        // two districts.
                        random_spanning_tree(&subgraph_buf.graph, &mut mst_buf, &mut range_buf, &mut rng);
                        // Step 3: choose a random balance edge, if possible.
                        let split = random_split(
                            &subgraph_buf.graph,
                            &mut rng,
                            &mst_buf.mst,
                            dist_a,
                            dist_b,
                            &mut split_buf,
                            &mut proposal_buf,
                            &subgraph_buf.raw_nodes,
                            &params,
                        );
                        match split {
                            Ok(n_splits) => {
                                // Step 4: accept any particular edge with probability 1 / (M * seam length)
                                let seam_length = proposal_buf.seam_length(&graph);
                                let prob = (n_splits as f64) / (seam_length as f64 * params.M as f64);
                                if prob > 1 {
                                    panic!("Invalid state: got {} splits, seam length {}", n_splits, seam_length);
                                }
                                if rng.gen::<f64>() < prob {
                                    proposals.push(proposal_buf.clone());
                                } else {
                                    counts.seam_length += 1;
                                }
                            }
                            Err(_) => counts.no_split += 1, // TODO: break out errors?
                        }
                    }
                    res_s.send(ResultPacket {
                        counts: counts,
                        proposals: proposals,
                    });
                    next = job_r.recv().unwrap();
                }
            });
        }

        if params.num_steps > 0 {
            for job in job_sends.iter() {
                job.send(JobPacket{ n_steps: batch_size, diff: None, terminate: false });
            }
        }

        print!("step\tnon_adjacent\tno_split\tseam_length\ta_label\tb_label\t");
        println!("a_pop\tb_pop\ta_nodes\tb_nodes");
        while step <= params.num_steps {
            let mut counts = ChainCounts::default();
            let mut proposals = Vec::<RecomProposal>::new();
            for i in 0..n_threads {
                let packet: ResultPacket = result_recv.recv().unwrap();
                counts = counts + packet.counts;
                proposals.extend(packet.proposals);
            }
            let loops = counts.non_adjacent + counts.no_split + counts.seam_length;
            if proposals.len() > 0 {
                // Sample events without replacement.
                let mut total = loops + proposals.len();
                while total > 0 {
                    let event = rng.gen_range(0..total);
                    if event >= 0 && event < counts.non_adjacent {
                        state.non_adjacent += 1;
                        counts.non_adjacent -= 1;
                    } else if event >= counts.non_adjacent && event < counts.non_adjacent + counts.no_split {
                        state.no_split += 1;
                        counts.no_split -= 1;
                    } else if event >= counts.non_adjacent + counts.no_split && event < counts.non_adjacent + counts.no_split + counts.seam_length {
                        state.seam_length += 1;
                        counts.seam_length -= 1;
                    } else {
                        let proposal = &proposals[rng.gen_range(0..proposals.len())];
                        for job in job_sends.iter() {
                            job.send(JobPacket {
                                n_steps: batch_size,
                                diff: Some(proposal.clone()),
                                terminate: false
                            });
                        }
                        println!("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:?}\t{:?}",
                                 step, counts.non_adjacent, counts.no_split, counts.seam_length,
                                 proposal.a_label, proposal.b_label,
                                 proposal.a_pop, proposal.b_pop,
                                 proposal.a_nodes, proposal.b_nodes);
                        break;
                    }
                    total -= 1;
                    step += 1;
                }
            } else {
                state = state + counts;
                step += loops as u64;
                for job in job_sends.iter() {
                    job.send(JobPacket{ n_steps: batch_size, diff: None, terminate: false });
                }
            }
        }
        for job in job_sends.iter() {
            job.send(JobPacket{ n_steps: batch_size, diff: None, terminate: true });
        }
    }).unwrap();
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
        .arg(
            Arg::with_name("n_threads")
                .long("n-threads")
                .takes_value(true)
                .required(true)
                .help("The number of threads to use.")
        )
        .arg(
            Arg::with_name("batch_size")
                .long("batch-size")
                .takes_value(true)
                .required(true)
                .help("The number of proposals per batch job.")
        )
        .get_matches();
    let n_steps = value_t!(matches.value_of("n_steps"), u64).unwrap_or_else(|e| e.exit());
    let rng_seed = value_t!(matches.value_of("rng_seed"), u64).unwrap_or_else(|e| e.exit());
    let tol = value_t!(matches.value_of("tol"), f64).unwrap_or_else(|e| e.exit());
    let M = value_t!(matches.value_of("M"), u32).unwrap_or_else(|e| e.exit());
    let n_threads = value_t!(matches.value_of("n_threads"), usize).unwrap_or_else(|e| e.exit());
    let batch_size = value_t!(matches.value_of("batch_size"), usize).unwrap_or_else(|e| e.exit());
    let graph_json = fs::canonicalize(PathBuf::from(matches.value_of("graph_json").unwrap())).unwrap().into_os_string().into_string().unwrap();
    let pop_col = matches.value_of("pop_col").unwrap();
    let assignment_col = matches.value_of("assignment_col").unwrap();

    assert!(tol >= 0.0 && tol <= 1.0);

    let (graph, mut partition) = from_networkx(
        &graph_json,
        pop_col,
        assignment_col
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

    let mut graph_file = fs::File::open(&graph_json).unwrap();
    let mut graph_hasher = Sha3_256::new();
    io::copy(&mut graph_file, &mut graph_hasher);
    let graph_hash = format!("{:x}", graph_hasher.finalize());
    let meta = json!({
        "M": M,
        "assignment_col": assignment_col,
        "tol": tol,
        "pop_col": pop_col,
        "graph_path": graph_json,
        "graph_sha3": graph_hash,
        "batch_size": batch_size,
        "rng_seed": rng_seed,
        "num_threads": n_threads,
        "num_steps": n_steps,
        "parallel": true,
       "graph_json": graph_json
    });
    println!("{}", meta.to_string());
    run_multi_chain(&graph, &partition, params, n_threads, batch_size);
    //println!("done");
}
