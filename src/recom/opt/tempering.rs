//! ReCom-based optimization using parallel tempering.
use super::super::{
    node_bound, random_split, uniform_dist_pair, RecomParams, RecomProposal, RecomVariant,
};
use super::{Optimizer, ScoreValue};
use crate::buffers::{SpanningTreeBuffer, SplitBuffer, SubgraphBuffer};
use crate::graph::Graph;
use crate::partition::Partition;
use crate::spanning_tree::{RMSTSampler, RegionAwareSampler, SpanningTreeSampler};
use crossbeam::scope;
use crossbeam_channel::{unbounded, Receiver, Sender};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde_json::json;
use std::collections::HashMap;
use std::marker::Send;

/// A unit of multithreaded work.
struct OptJobPacket {
    /// The number of steps to sample (*not* the number of accepted steps).
    n_steps: usize,
    /// The partition to start the next chain segment with.
    start: Option<Partition>,
    /// The temperature of the chain segment.
    temperature: f64,
    /// A sentinel used to kill the worker thread.
    terminate: bool,
}

/// The result of a unit of multithreaded work.
struct OptResultPacket {
    /// The best proposal found in a unit of work according to an
    /// objective function.
    best_partition: Option<Partition>,
    /// The score of the best proposal.
    best_score: Option<ScoreValue>,
    /// The last accepted proposal found in a unit of work according
    /// to an objective function.
    last_partition: Partition,
    /// The score of the last proposal.
    last_score: ScoreValue,
    /// The temperature used for the completed chain segment.
    temperature: f64,
}

/// Starts a ReCom optimization thread.
///
/// Arguments:
/// * `graph` - The graph associated with the chain.
/// * `partition` - The initial state of the chain.
/// * `params` - The chain parameters.
/// * `obj_fn` - The objective function to evaluate proposals against.
/// * `rng_seed` - The RNG seed for the job thread. (This should differ across threads.)
/// * `buf_size` - The buffer size for various chain buffers. This should usually be twice
///   the maximum possible district size (in nodes).
/// * `job_recv` - A Crossbeam channel for receiving batches of work from the main thread.
/// * `result_send` - A Crossbeam channel for sending completed batches to the main thread.
fn start_opt_thread(
    graph: Graph,
    params: RecomParams,
    obj_fn: impl Fn(&Graph, &Partition) -> ScoreValue + Send + Copy,
    rng_seed: u64,
    buf_size: usize,
    job_recv: Receiver<OptJobPacket>,
    result_send: Sender<OptResultPacket>,
) {
    let n = graph.pops.len();
    let mut rng: SmallRng = SeedableRng::seed_from_u64(rng_seed);
    let mut subgraph_buf = SubgraphBuffer::new(n, buf_size);
    let mut st_buf = SpanningTreeBuffer::new(buf_size);
    let mut split_buf = SplitBuffer::new(buf_size, params.balance_ub as usize);
    let mut proposal_buf = RecomProposal::new_buffer(buf_size);
    let mut st_sampler: Box<dyn SpanningTreeSampler>;
    if params.variant == RecomVariant::DistrictPairsRegionAware {
        st_sampler = Box::new(RegionAwareSampler::new(
            buf_size,
            params.region_weights.clone().unwrap(),
        ));
    } else if params.variant == RecomVariant::DistrictPairsRMST {
        st_sampler = Box::new(RMSTSampler::new(buf_size));
    } else {
        panic!("ReCom variant not supported by optimizer.");
    }

    let mut next: OptJobPacket = job_recv.recv().unwrap();
    while !next.terminate {
        let mut partition = next.start.unwrap();

        let mut best_partition: Option<Partition> = None;
        let mut score = obj_fn(&graph, &partition);
        let mut best_score: ScoreValue = score;
        let mut step = 0;

        while step < next.n_steps {
            // Sample a ReCom step.
            let dist_pair = uniform_dist_pair(&graph, &mut partition, &mut rng);
            if dist_pair.is_none() {
                continue;
            }
            let (dist_a, dist_b) = dist_pair.unwrap();
            partition.subgraph_with_attr(&graph, &mut subgraph_buf, dist_a, dist_b);
            st_sampler.random_spanning_tree(&subgraph_buf.graph, &mut st_buf, &mut rng);
            let split = random_split(
                &subgraph_buf.graph,
                &mut rng,
                &st_buf.st,
                dist_a,
                dist_b,
                &mut split_buf,
                &mut proposal_buf,
                &subgraph_buf.raw_nodes,
                &params,
            );
            if split.is_ok() {
                let mut next_partition = partition.clone();
                next_partition.update(&proposal_buf);
                let next_score = obj_fn(&graph, &next_partition);
                if next_score >= score {
                    score = next_score;
                    partition = next_partition;
                    if next_score >= best_score {
                        best_partition = Some(partition.clone());
                        best_score = next_score;
                    }
                } else {
                    let score_diff = (next_score - score) / next.temperature;
                    let prob = score_diff.exp();
                    if rng.gen::<f64>() < prob {
                        // Accept with some probability.
                        score = next_score;
                        partition = next_partition;
                    }
                }
                step += 1;
            }
        }
        let result = match best_partition {
            Some(ref partition) => OptResultPacket {
                best_partition: best_partition.clone(),
                best_score: Some(best_score),
                last_partition: partition.clone(),
                last_score: score,
                temperature: next.temperature,
            },
            None => OptResultPacket {
                best_partition: None,
                best_score: None,
                last_partition: partition,
                last_score: score,
                temperature: next.temperature,
            },
        };
        result_send.send(result).unwrap();
        next = job_recv.recv().unwrap();
    }
}

/// Sends a batch of work to a ReCom optimization thread.
fn next_batch(send: &Sender<OptJobPacket>, n_steps: usize, start: Partition, temperature: f64) {
    send.send(OptJobPacket {
        n_steps: n_steps,
        start: Some(start),
        temperature: temperature,
        terminate: false,
    })
    .unwrap();
}

/// Stops a ReCom optimization thread.
fn stop_opt_thread(send: &Sender<OptJobPacket>) {
    send.send(OptJobPacket {
        n_steps: 0,
        start: None,
        temperature: 0.0,
        terminate: true,
    })
    .unwrap();
}

#[derive(Clone)]
pub struct ParallelTemperingOptimizer {
    /// Chain parameters.
    params: RecomParams,
    /// Parallel chain temperatures (determines number of worker threads).
    temps: Vec<f64>,
    /// Number of steps per chain per probabilistic swap event.
    steps_per_swap: usize,
    /// Print the best intermediate results?
    verbose: bool,
}

impl ParallelTemperingOptimizer {
    pub fn new(
        params: RecomParams,
        temps: Vec<f64>,
        steps_per_swap: usize,
        verbose: bool,
    ) -> ParallelTemperingOptimizer {
        ParallelTemperingOptimizer {
            params,
            temps,
            steps_per_swap,
            verbose,
        }
    }
}

impl Optimizer for ParallelTemperingOptimizer {
    /// Runs a multi-threaded ReCom parallel tempering optimizer.
    ///
    /// # Arguments
    ///
    /// * `graph` - The graph associated with `partition`.
    /// * `partition` - The partition to start the chain run from (updated in place).
    /// * `obj_fn` - The objective to maximize.
    fn optimize(
        &self,
        graph: &Graph,
        partition: Partition,
        obj_fn: impl Fn(&Graph, &Partition) -> ScoreValue + Send + Clone + Copy,
    ) -> Partition {
        let mut step = 0;
        let node_ub = node_bound(&graph.pops, self.params.max_pop);
        let n_threads = self.temps.len();
        let mut job_sends = vec![]; // main thread sends work to job threads
        let mut job_recvs = vec![]; // job threads receive work from main thread
        let mut rng: SmallRng = SeedableRng::seed_from_u64(self.params.rng_seed);
        for _ in 0..n_threads {
            let (s, r): (Sender<OptJobPacket>, Receiver<OptJobPacket>) = unbounded();
            job_sends.push(s);
            job_recvs.push(r);
        }
        // All optimization threads send a summary of chain results back to the main thread.
        let (result_send, result_recv): (Sender<OptResultPacket>, Receiver<OptResultPacket>) =
            unbounded();
        let init_score = obj_fn(&graph, &partition);
        let mut best_score = init_score;
        let mut best_partition = partition.clone();
        let mut best_cut_edges = best_partition.cut_edges(graph).len();

        scope(|scope| {
            // Start optimization threads.
            for t_idx in 0..n_threads {
                // TODO: is this (+ t_idx) a sensible way to seed?
                let rng_seed = self.params.rng_seed + t_idx as u64 + 1;
                let job_recv = job_recvs[t_idx].clone();
                let result_send = result_send.clone();

                scope.spawn(move |_| {
                    start_opt_thread(
                        graph.clone(),
                        self.params.clone(),
                        obj_fn,
                        rng_seed,
                        node_ub,
                        job_recv,
                        result_send
                    );
                });
            }

            if self.params.num_steps > 0 {
                for (job, temperature) in job_sends.iter().zip(self.temps.iter()) {
                    next_batch(job, self.steps_per_swap, partition.clone(), *temperature);
                }
            }

            let mut unimproved_cycles = 0;
            while step <= self.params.num_steps {
                let mut last_steps = vec![];
                let mut improved = false;
                for _ in 0..n_threads {
                    let packet: OptResultPacket = result_recv.recv().unwrap();
                    last_steps.push((packet.last_partition, packet.last_score, packet.temperature));
                    if let Some(cand_partition) = packet.best_partition {
                        //let cand_cut_edges = cand_partition.clone().cut_edges(graph).len();
                        let cand_score = packet.best_score.unwrap();
                        //if cand_score >= 4.0 && (cand_score > best_score || cand_cut_edges < best_cut_edges || cand_score >= 5.0) {
                        if cand_score > best_score {
                            best_score = packet.best_score.unwrap();
                            best_partition = cand_partition;
                            // best_cut_edges = cand_cut_edges;
                            improved = true;
                            if self.verbose {
                                println!("{}", json!({
                                    "step": step,
                                    "type": "opt",
                                    "score": best_score,
                                    "assignment": best_partition.assignments.clone().into_iter().enumerate().collect::<HashMap<usize, u32>>()
                                }).to_string());
                            }
                        }
                    }
                }
                step += (n_threads * self.steps_per_swap) as u64;

                if !improved {
                    unimproved_cycles += 1;
                }
                if unimproved_cycles > 1000 {
                    // Aggressive exchange: everyone gets the best plan.
                    for (idx, temperature) in self.temps.iter().enumerate() {
                        last_steps[idx] = (best_partition.clone(), best_score, *temperature);
                    }
                    unimproved_cycles = 0;
                } else {
                    // Probabilistic replica exchange: swap an adjacent temperature pair at random.
                    last_steps.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
                    let r_idx = rng.gen_range(0..n_threads - 1);
                    let d_inv_temp = (1.0 / last_steps[r_idx].2) - (1.0 / last_steps[r_idx + 1].2);
                    let d_energy = last_steps[r_idx + 1].1 - last_steps[r_idx].1;
                    let prob_swap = (d_inv_temp * d_energy).exp();
                    if rng.gen::<f64>() < prob_swap {
                        let (partition_a, score_a, temp_a) = last_steps[r_idx].clone();
                        let (partition_b, score_b, temp_b) = last_steps[r_idx + 1].clone();
                        last_steps[r_idx] = (partition_b, score_b, temp_a);
                        last_steps[r_idx + 1] = (partition_a, score_a, temp_b);
                    }
                }

                for (job, (partition, _, temperature)) in job_sends.iter().zip(last_steps.iter()) {
                    next_batch(job, self.steps_per_swap, partition.clone(), *temperature);
                }
            }

            // Terminate worker threads.
            for job in job_sends.iter() {
                stop_opt_thread(job);
            }
            best_partition
        })
        .unwrap()
    }
}
