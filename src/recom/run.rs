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
use crossbeam_channel::{bounded, unbounded};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Determines how many proposals the stats thread can lag behind by
/// (compared to the head of the chain).
const STATS_CHANNEL_CAPACITY: usize = 16;

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

/// Information necessary to compute statistics about an accepted proposal.
struct StepPacket {
    /// The current step count of the chain.
    step: u64,
    /// The accepted proposal (only `None` when the termination sentinel is set.)
    proposal: Option<RecomProposal>,
    /// The self-loop counts leading up to the proposal.
    counts: SelfLoopCounts,
    /// A sentinel used to kill the worker thread.
    terminate: bool,
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
    let mut job_sends = vec![]; // main thread sends work to job threads
    let mut job_recvs = vec![]; // job threads receive work from main thread
    for _ in 0..n_threads {
        // TODO: fancy unzipping?
        let (s, r) = unbounded();
        job_sends.push(s);
        job_recvs.push(r);
    }
    // All job threads send a summary of chain results back to the main thread.
    let (result_send, result_recv) = unbounded();
    // The stats thread receives accepted proposals from the main thread.
    let (stats_send, stats_recv) = bounded(STATS_CHANNEL_CAPACITY);
    let mut rng: SmallRng = SeedableRng::seed_from_u64(params.rng_seed);
    let reversible = params.variant == RecomVariant::Reversible;
    let sample_district_pairs = reversible
        || params.variant == RecomVariant::DistrictPairsUST
        || params.variant == RecomVariant::DistrictPairsRMST;
    let rmst = params.variant == RecomVariant::CutEdgesRMST
        || params.variant == RecomVariant::DistrictPairsRMST;

    // Start job and stats threads.
    scope(|scope| {
        // Start stats thread.
        {
            let graph = graph.clone();
            let mut partition = partition.clone();
            scope.spawn(move |_| {
                writer.init(&graph, &partition).unwrap();
                let mut next: StepPacket = stats_recv.recv().unwrap();
                while !next.terminate {
                    let proposal = next.proposal.unwrap();
                    partition.update(&proposal);
                    writer
                        .step(next.step, &graph, &partition, &proposal, &next.counts)
                        .unwrap();
                    next = stats_recv.recv().unwrap();
                }
                writer.close().unwrap();
            });
        }

        // Start job threads.
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
                    st_sampler = Box::new(RMSTSampler::new(node_ub));
                } else {
                    st_sampler = Box::new(USTSampler::new(node_ub, &mut rng));
                }

                let mut next: JobPacket = job_r.recv().unwrap();
                while !next.terminate {
                    match next.diff {
                        Some(diff) => partition.update(&diff),
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
                            let num_dists = partition.num_dists;
                            let dist_adj = partition.dist_adj(&graph);
                            if dist_adj[(dist_a * num_dists as usize) + dist_b] == 0 {
                                counts.inc(SelfLoopReason::NonAdjacent);
                                continue;
                            }
                        } else {
                            // Sample a cut edge, which is guaranteed to yield a pair of adjacent districts.
                            let cut_edges = partition.cut_edges(&graph);
                            let cut_edge_idx = rng.gen_range(0..cut_edges.len()) as usize;
                            let edge_idx = cut_edges[cut_edge_idx] as usize;
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
                        // Case: no accepted proposal (don't need to update worker thread state).
                        sampled.inc(counts.index_and_dec(event).unwrap());
                        loops -= 1;
                    } else {
                        // Case: accepted proposal (update worker thread state).
                        let proposal = &proposals[rng.gen_range(0..proposals.len())];
                        for job in job_sends.iter() {
                            job.send(JobPacket {
                                n_steps: batch_size,
                                diff: Some(proposal.clone()),
                                terminate: false,
                            })
                            .unwrap();
                        }
                        stats_send
                            .send(StepPacket {
                                step: step,
                                proposal: Some(proposal.clone()),
                                counts: sampled,
                                terminate: false,
                            })
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

        // Terminate worker threads.
        for job in job_sends.iter() {
            job.send(JobPacket {
                n_steps: batch_size,
                diff: None,
                terminate: true,
            })
            .unwrap();
        }
        stats_send
            .send(StepPacket {
                step: step,
                proposal: None,
                counts: SelfLoopCounts::default(),
                terminate: true,
            })
            .unwrap();
    })
    .unwrap();
}
