//! ReCom-based optimization using short bursts.
//!
//! We use the "short bursts" heuristic introduced in Cannon et al. 2020
//! (see "Voting Rights, Markov Chains, and Optimization by Short Bursts",
//!  arXiv: 2011.02288) to maximize arbitrary partition-level objective
//! functions.
use super::super::{
    node_bound, random_split, uniform_dist_pair, RecomParams, RecomProposal, RecomVariant,
};
use super::{Optimizer, ScoreValue};
use crate::buffers::{SpanningTreeBuffer, SplitBuffer, SubgraphBuffer};
use crate::graph::Graph;
use crate::partition::Partition;
use crate::spanning_tree::{RMSTSampler, RegionAwareSampler, SpanningTreeSampler};
use crate::stats::SelfLoopCounts;
use crate::stats::{JSONLTwoLineWriter, StatsWriter};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::sync::Arc;

pub struct VerboseBurstsOptimizer {
    /// Chain parameters.
    params: RecomParams,
    /// The number of steps per burst.
    burst_length: usize,
}

impl VerboseBurstsOptimizer {
    pub fn new(params: RecomParams, burst_length: usize) -> VerboseBurstsOptimizer {
        VerboseBurstsOptimizer {
            params: params,
            burst_length: burst_length,
        }
    }
}

impl Optimizer for VerboseBurstsOptimizer {
    /// Runs a multi-threaded ReCom short bursts optimizer.
    ///
    /// # Arguments
    ///
    /// * `graph` - The graph associated with `partition`.
    /// * `partition` - The partition to start the chain run from (updated in place).
    /// * `obj_fn` - The objective to maximize.
    fn optimize(
        &self,
        graph: &Graph,
        mut partition: Partition,
        obj_fn: Arc<dyn Fn(&Graph, &Partition) -> ScoreValue + Send + Sync>,
    ) -> Partition {
        let mut step = 0;
        let buf_size = node_bound(&graph.pops, self.params.max_pop);
        let n = graph.pops.len();
        let mut rng: SmallRng = SeedableRng::seed_from_u64(self.params.rng_seed);
        let mut subgraph_buf = SubgraphBuffer::new(n, buf_size);
        let mut st_buf = SpanningTreeBuffer::new(buf_size);
        let mut split_buf = SplitBuffer::new(buf_size, self.params.balance_ub as usize);
        let mut proposal_buf = RecomProposal::new_buffer(buf_size);
        let mut st_sampler: Box<dyn SpanningTreeSampler>;
        let mut writer = JSONLTwoLineWriter::new();

        if self.params.variant == RecomVariant::DistrictPairsRegionAware {
            st_sampler = Box::new(RegionAwareSampler::new(
                buf_size,
                self.params.region_weights.clone().unwrap(),
            ));
        } else if self.params.variant == RecomVariant::DistrictPairsRMST {
            st_sampler = Box::new(RMSTSampler::new(buf_size));
        } else {
            panic!("ReCom variant not supported by optimizer.");
        }

        writer.init(&graph, &partition).unwrap();
        while step <= self.params.num_steps {
            let mut best_partition = partition.clone();
            let mut score = obj_fn(&graph, &partition);
            let mut best_score: ScoreValue = score;
            let mut burst_step = 0;

            while burst_step < self.burst_length {
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
                    &self.params,
                );
                if split.is_ok() {
                    score = obj_fn(&graph, &partition);
                    partition.update(&proposal_buf);
                    if score >= best_score {
                        best_partition = partition.clone();
                        best_score = score;
                    }

                    writer
                        .step(
                            step + burst_step as u64,
                            &graph,
                            &partition,
                            &proposal_buf,
                            &SelfLoopCounts::default(),
                        )
                        .unwrap();
                    burst_step += 1;
                }
            }
            partition = best_partition.clone();
            step += burst_step as u64;
        }
        partition
    }
}
