//! ReCom-based heuristic optimization algorithms.
use super::{Graph, Partition};

pub type ScoreValue = f64;

pub trait Optimizer {
    fn optimize(
        &self,
        graph: &Graph,
        partition: Partition,
        obj_fn: impl Fn(&Graph, &Partition) -> ScoreValue + Send + Clone + Copy,
    ) -> Partition;
}

mod short_bursts;
mod tempering;
pub use short_bursts::ShortBurstsOptimizer;
pub use tempering::ParallelTemperingOptimizer;
