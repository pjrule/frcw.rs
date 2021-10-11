//! Runners for ReCom.
//!
//! A runner orchestrates the various components of the ReCom algorithm
//! (spanning tree generation, etc.) and handles setup, output, the
//! collection of auxiliary statistics, and (optionally) multithreading.
//!
//! Currently, there is only one runner ([`multi_chain`]). This runner
//! is multithreaded and prints accepted proposals to `stdout` in TSV format.
//! It also collects rejection/self-loop statistics.
use super::{
    cut_edge_dist_pair, node_bound, random_split, uniform_dist_pair, RecomParams, RecomProposal,
    RecomVariant,
};
use crate::buffers::{SpanningTreeBuffer, SplitBuffer, SubgraphBuffer};
use crate::graph::Graph;
use crate::partition::Partition;
use crate::spanning_tree::{RMSTSampler, RegionAwareSampler, SpanningTreeSampler, USTSampler};
use crate::stats::{SelfLoopCounts, SelfLoopReason, StatsWriter};
use crossbeam::scope;
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Determines how many proposals the stats thread can lag behind by
/// (compared to the head of the chain).
const STATS_CHANNEL_CAPACITY: usize = 16;

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

/// Starts a thread that writes statistics from accepted plans to `stdout`.
fn start_stats_thread(
    graph: Graph,
    mut partition: Partition,
    mut writer: Box<dyn StatsWriter>,
    recv: Receiver<StepPacket>,
) {
    writer.init(&graph, &partition).unwrap();
    let mut next: StepPacket = recv.recv().unwrap();
    while !next.terminate {
        let proposal = next.proposal.unwrap();
        partition.update(&proposal);
        writer
            .step(next.step, &graph, &partition, &proposal, &next.counts)
            .unwrap();
        next = recv.recv().unwrap();
    }
    writer.close().unwrap();
}

/// Stops a statistics writer thread.
fn stop_stats_thread(send: &Sender<StepPacket>) {
    send.send(StepPacket {
        step: 0,
        proposal: None,
        counts: SelfLoopCounts::default(),
        terminate: true,
    })
    .unwrap();
}

/// Starts a ReCom job thread.
/// ReCom job threads sample batches of proposals, which are then aggregated by
/// the main thread. (Thus, this function contains most of the ReCom chain logic.)
///
/// Arguments:
/// * `graph` - The graph associated with the chain.
/// * `partition` - The initial state of the chain.
/// * `params` - The chain parameters.
/// * `rng_seed` - The RNG seed for the job thread. (This should differ across threads.)
/// * `buf_size` - The buffer size for various chain buffers. This should usually be twice
///   the maximum possible district size (in nodes).
/// * `job_recv` - A Crossbeam channel for receiving batches of work from the main thread.
/// * `result_send` - A Crossbeam channel for sending completed batches to the main thread.
fn start_job_thread(
    graph: Graph,
    mut partition: Partition,
    params: RecomParams,
    rng_seed: u64,
    buf_size: usize,
    job_recv: Receiver<JobPacket>,
    result_send: Sender<ResultPacket>,
) {
    let n = graph.pops.len();
    let mut rng: SmallRng = SeedableRng::seed_from_u64(rng_seed);
    let mut subgraph_buf = SubgraphBuffer::new(n, buf_size);
    let mut st_buf = SpanningTreeBuffer::new(buf_size);
    let mut split_buf = SplitBuffer::new(buf_size, params.balance_ub as usize);
    let mut proposal_buf = RecomProposal::new_buffer(buf_size);
    let mut st_sampler: Box<dyn SpanningTreeSampler>;

    let reversible = params.variant == RecomVariant::Reversible;
    let sample_district_pairs = reversible
        || params.variant == RecomVariant::DistrictPairsUST
        || params.variant == RecomVariant::DistrictPairsRMST
        || params.variant == RecomVariant::DistrictPairsRegionAware;
    let rmst = params.variant == RecomVariant::CutEdgesRMST
        || params.variant == RecomVariant::DistrictPairsRMST
        || params.variant == RecomVariant::DistrictPairsRegionAware
        || params.variant == RecomVariant::CutEdgesRMST
        || params.variant == RecomVariant::CutEdgesRegionAware;
    let region_aware = params.variant == RecomVariant::CutEdgesRegionAware
        || params.variant == RecomVariant::DistrictPairsRegionAware;

    let mut region_aware_attrs: Vec<String> = vec![];
    if region_aware {
        st_sampler = Box::new(RegionAwareSampler::new(
            buf_size,
            params.region_weights.clone().unwrap(),
        ));
        region_aware_attrs = params
            .region_weights
            .clone()
            .unwrap()
            .iter()
            .map(|(col, _)| col.to_owned())
            .collect();
    } else if rmst {
        st_sampler = Box::new(RMSTSampler::new(buf_size));
    } else {
        st_sampler = Box::new(USTSampler::new(buf_size, &mut rng));
    }

    let mut next: JobPacket = job_recv.recv().unwrap();
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
                // Sample a pair of districts uniformly at random, self-looping if an
                // adjacent pair is not found.
                match uniform_dist_pair(&graph, &mut partition, &mut rng) {
                    Some((a, b)) => {
                        dist_a = a;
                        dist_b = b;
                    }
                    None => {
                        counts.inc(SelfLoopReason::NonAdjacent);
                        continue;
                    }
                }
            } else {
                // Sample a cut edge, which is guaranteed to yield a pair of adjacent districts.
                let (a, b) = cut_edge_dist_pair(&graph, &mut partition, &mut rng);
                dist_a = a;
                dist_b = b;
            }
            if region_aware {
                // Region-aware ReCom requires extra node-level metadata
                // (region assignments, e.g. county IDs).
                partition.subgraph_with_attr_subset(
                    &graph,
                    &mut subgraph_buf,
                    region_aware_attrs.iter(),
                    dist_a,
                    dist_b,
                );
            } else {
                partition.subgraph(&graph, &mut subgraph_buf, dist_a, dist_b);
            }

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
                        let prob =
                            (n_splits as f64) / (seam_length as f64 * params.balance_ub as f64);
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
        result_send
            .send(ResultPacket {
                counts: counts,
                proposals: proposals,
            })
            .unwrap();
        next = job_recv.recv().unwrap();
    }
}

fn next_batch(send: &Sender<JobPacket>, diff: Option<RecomProposal>, batch_size: usize) {
    send.send(JobPacket {
        n_steps: batch_size,
        diff: diff,
        terminate: false,
    })
    .unwrap();
}

/// Stops a ReCom job thread.
fn stop_job_thread(send: &Sender<JobPacket>) {
    send.send(JobPacket {
        n_steps: 0,
        diff: None,
        terminate: true,
    })
    .unwrap();
}

/// Runs a multi-threaded ReCom chain.
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
    writer: Box<dyn StatsWriter>,
    params: &RecomParams,
    n_threads: usize,
    batch_size: usize,
) {
    let mut step = 0;
    let node_ub = node_bound(&graph.pops, params.max_pop);
    let mut job_sends = vec![]; // main thread sends work to job threads
    let mut job_recvs = vec![]; // job threads receive work from main thread
    for _ in 0..n_threads {
        let (s, r): (Sender<JobPacket>, Receiver<JobPacket>) = unbounded();
        job_sends.push(s);
        job_recvs.push(r);
    }
    // All job threads send a summary of chain results back to the main thread.
    let (result_send, result_recv): (Sender<ResultPacket>, Receiver<ResultPacket>) = unbounded();
    // The stats thread receives accepted proposals from the main thread.
    let (stats_send, stats_recv): (Sender<StepPacket>, Receiver<StepPacket>) =
        bounded(STATS_CHANNEL_CAPACITY);
    let mut rng: SmallRng = SeedableRng::seed_from_u64(params.rng_seed);

    // Start job and stats threads.
    scope(|scope| {
        // Start stats thread.
        scope.spawn(move |_| {
            start_stats_thread(graph.clone(), partition.clone(), writer, stats_recv);
        });

        // Start job threads.
        for t_idx in 0..n_threads {
            // TODO: is this (+ t_idx) a sensible way to seed?
            let rng_seed = params.rng_seed + t_idx as u64 + 1;
            let job_recv = job_recvs[t_idx].clone();
            let result_send = result_send.clone();

            scope.spawn(move |_| {
                start_job_thread(
                    graph.clone(),
                    partition.clone(),
                    params.clone(),
                    rng_seed,
                    node_ub,
                    job_recv,
                    result_send,
                );
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
                            next_batch(job, Some(proposal.clone()), batch_size);
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
                    next_batch(job, None, batch_size);
                }
            }
        }

        // Terminate worker threads.
        for job in job_sends.iter() {
            stop_job_thread(job);
        }
    })
    .unwrap();
    stop_stats_thread(&stats_send);
}
