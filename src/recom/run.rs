//! Runners for ReCom.
//!
//! A runner orchestrates the various components of the ReCom algorithm
//! (spanning tree generation, etc.) and handles setup, output, the
//! collection of auxiliary statistics, and (optionally) multithreading.
//!
//! Currently, there is only one runner ([`multi_chain`]). This runner
//! is multithreaded and prints accepted proposals to `stdout` in TSV format.
//! It also collects rejection/self-loop statistics.
use super::{random_split, RecomParams, RecomProposal, RecomVariant};
use crate::buffers::{SpanningTreeBuffer, SplitBuffer, SubgraphBuffer};
use crate::graph::Graph;
use crate::partition::Partition;
use crate::spanning_tree::{RMSTSampler, SpanningTreeSampler, USTSampler};
use crate::stats::{SelfLoopCounts, SelfLoopReason, StatsWriter};
use crossbeam::scope;
use crossbeam_channel::unbounded;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Returns the maximum number of nodes in two districts based on node
/// populations (`pop`) and the maximum district population (`max_pop`).
///
/// Used to choose buffer sizes for recombination steps.
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

/// A unit of multithreaded work.
struct JobPacket {
    /// The number of steps to sample (*not* the number of unique plans).
    n_steps: usize,
    /// The change in the chain state since the last batch of work.
    /// If no new proposal is accepted, this may be `None`.
    diff: Option<RecomProposal>,
    /// A sentinel used to kill the worker thread.
    terminate: bool,
}

/// The result of a unit of multithreaded work.
struct ResultPacket {
    /// Self-loop statistics.
    counts: SelfLoopCounts,
    /// ≥0 valid proposals generated within the unit of work.
    proposals: Vec<RecomProposal>,
}

/// Runs a multi-threaded ReCom chain and prints accepted
/// proposals to `stdout` in TSV format.
///
/// Rows in the output contain the following columns:
///   * `step` - The step count at the accepted proposal (including self-loops).
///   * `non_adjacent` - The number of self-loops due to non-adjacency.
///   * `no_split` - The number of self-loops due to the lack of an ε-balanced split.
///   * `seam_length` - The number of self-loops due to seam length rejection
///     (Reversible ReCom only).
///   * `a_label` - The label of the `a`-district in the proposal.
///   * `b_label` - The label of the `b`-district in the proposal.
///   * `a_pop` - The population of the new `a`-district.
///   * `b_pop` - The population of the new `b`-district.
///   * `a_nodes` - The list of node indices in the new `a`-district.
///   * `b_nodes` - The list of node indices in the new `b`-district.
///
/// # Arguments
///
/// * `graph` - The graph associated with `partition`.
/// * `partition` - The partition to start the chain run from (updated in place).
/// * `writer` - The statistics writer.
/// * `params` - The parameters of the ReCom chain run.
/// * `n_threads` - The number of worker threads (excluding the main thread).
/// * `batch_size` - The number of steps per unit of multithreaded work. This
///   parameter should be tuned according to the chain's average acceptance
///   probability: chains that reject most proposals (e.g. reversible ReCom
///   on large graphs) will benefit from large batches, but chains that accept
///   most or all proposals should use small batches.
pub fn multi_chain(
    graph: &Graph,
    partition: &Partition,
    mut writer: Box<dyn StatsWriter>,
    params: RecomParams,
    n_threads: usize,
    batch_size: usize,
) {
    let mut step = 0;
    let node_ub = node_bound(&graph.pops, params.max_pop);
    let mut job_sends = vec![];
    let mut job_recvs = vec![];
    for _ in 0..n_threads {
        // TODO: fancy unzipping?
        let (s, r) = unbounded();
        job_sends.push(s);
        job_recvs.push(r);
    }
    let (result_send, result_recv) = unbounded();
    let mut rng: SmallRng = SeedableRng::seed_from_u64(params.rng_seed);
    let reversible = params.variant == RecomVariant::Reversible;
    let sample_district_pairs = reversible
        || params.variant == RecomVariant::DistrictPairsUST
        || params.variant == RecomVariant::DistrictPairsRMST;
    let rmst = params.variant == RecomVariant::CutEdgesRMST
        || params.variant == RecomVariant::DistrictPairsRMST;

    // Start threads.
    scope(|scope| {
        for t_idx in 0..n_threads {
            let job_r = job_recvs[t_idx].clone();
            let res_s = result_send.clone();
            let graph = graph.clone();
            let mut partition = partition.clone();
            scope.spawn(move |_| {
                let n = graph.pops.len();
                // TODO: is this (+ t_idx) a sensible way to seed?
                let mut rng: SmallRng =
                    SeedableRng::seed_from_u64(params.rng_seed + t_idx as u64 + 1);
                let mut subgraph_buf = SubgraphBuffer::new(n, node_ub);
                let mut st_buf = SpanningTreeBuffer::new(node_ub);
                let mut split_buf = SplitBuffer::new(node_ub, params.balance_ub as usize);
                let mut proposal_buf = RecomProposal::new_buffer(node_ub);
                let mut st_sampler: Box<dyn SpanningTreeSampler>;
                if rmst {
                    st_sampler = Box::new(RMSTSampler::new());
                } else {
                    st_sampler = Box::new(USTSampler::new(&mut rng));
                }

                let mut next: JobPacket = job_r.recv().unwrap();
                while !next.terminate {
                    match next.diff {
                        Some(diff) => partition.update(&graph, &diff),
                        None => {}
                    }
                    let mut counts = SelfLoopCounts::default();
                    let mut proposals = Vec::<RecomProposal>::new();
                    for _ in 0..next.n_steps {
                        // Step 1: sample a pair of adjacent districts.
                        let (dist_a, dist_b);
                        if sample_district_pairs {
                            // Choose district pairs at random until finding an adjacent pair.
                            dist_a = rng.gen_range(0..partition.num_dists) as usize;
                            dist_b = rng.gen_range(0..partition.num_dists) as usize;
                            if partition.dist_adj[(dist_a * partition.num_dists as usize) + dist_b]
                                == 0
                            {
                                counts.inc(SelfLoopReason::NonAdjacent);
                                continue;
                            }
                        } else {
                            // Sample a cut edge, which is guaranteed to yield a pair of adjacent districts.
                            let cut_edge_idx = rng.gen_range(0..partition.cut_edges.len()) as usize;
                            let edge_idx = partition.cut_edges[cut_edge_idx] as usize;
                            dist_a = partition.assignments[graph.edges[edge_idx].0] as usize;
                            dist_b = partition.assignments[graph.edges[edge_idx].1] as usize;
                        }
                        partition.subgraph(&graph, &mut subgraph_buf, dist_a, dist_b);

                        // Step 2: draw a random spanning tree of the subgraph induced by the
                        // two districts.
                        st_sampler.random_spanning_tree(&subgraph_buf.graph, &mut st_buf, &mut rng);

                        // Step 3: choose a random balance edge, if possible.
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
                        match split {
                            Ok(n_splits) => {
                                if reversible {
                                    // Step 4: accept any particular edge with probability 1 / (M * seam length)
                                    let seam_length = proposal_buf.seam_length(&graph);
                                    let prob = (n_splits as f64)
                                        / (seam_length as f64 * params.balance_ub as f64);
                                    if prob > 1.0 {
                                        panic!(
                                            "Invalid state: got {} splits, seam length {}",
                                            n_splits, seam_length
                                        );
                                    }
                                    if rng.gen::<f64>() < prob {
                                        proposals.push(proposal_buf.clone());
                                    } else {
                                        counts.inc(SelfLoopReason::SeamLength);
                                    }
                                } else {
                                    // Accept.
                                    proposals.push(proposal_buf.clone());
                                }
                            }
                            Err(_) => counts.inc(SelfLoopReason::NoSplit), // TODO: break out errors?
                        }
                    }
                    res_s
                        .send(ResultPacket {
                            counts: counts,
                            proposals: proposals,
                        })
                        .unwrap();
                    next = job_r.recv().unwrap();
                }
            });
        }

        if params.num_steps > 0 {
            for job in job_sends.iter() {
                job.send(JobPacket {
                    n_steps: batch_size,
                    diff: None,
                    terminate: false,
                })
                .unwrap();
            }
        }
        writer.init(graph, partition).unwrap();
        let mut sampled = SelfLoopCounts::default();
        while step <= params.num_steps {
            let mut counts = SelfLoopCounts::default();
            let mut proposals = Vec::<RecomProposal>::new();
            for _ in 0..n_threads {
                let packet: ResultPacket = result_recv.recv().unwrap();
                counts = counts + packet.counts;
                proposals.extend(packet.proposals);
            }
            let mut loops = counts.sum();
            if proposals.len() > 0 {
                // Sample events without replacement.
                let mut total = loops + proposals.len();
                while total > 0 {
                    step += 1;
                    let event = rng.gen_range(0..total);
                    if event < loops {
                        sampled.inc(counts.index_and_dec(event).unwrap());
                        loops -= 1;
                    } else {
                        let proposal = &proposals[rng.gen_range(0..proposals.len())];
                        for job in job_sends.iter() {
                            job.send(JobPacket {
                                n_steps: batch_size,
                                diff: Some(proposal.clone()),
                                terminate: false,
                            })
                            .unwrap();
                        }
                        writer
                            .step(step, &graph, &partition, &proposal, &sampled)
                            .unwrap();
                        // Reset sampled rejection stats until the next accepted step.
                        sampled = SelfLoopCounts::default();
                        break;
                    }
                    total -= 1;
                }
            } else {
                sampled = sampled + counts;
                step += loops as u64;
                for job in job_sends.iter() {
                    job.send(JobPacket {
                        n_steps: batch_size,
                        diff: None,
                        terminate: false,
                    })
                    .unwrap();
                }
            }
        }
        for job in job_sends.iter() {
            job.send(JobPacket {
                n_steps: batch_size,
                diff: None,
                terminate: true,
            })
            .unwrap();
        }
    })
    .unwrap();
    writer.close().unwrap();
}
