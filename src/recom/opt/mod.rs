//! ReCom-based heuristic optimization algorithms.
use super::{Graph, Partition};
use std::sync::Arc;

pub type ScoreValue = f64;

pub trait Optimizer {
    fn optimize(
        &self,
        graph: &Graph,
        partition: Partition,
        obj_fn: Arc<dyn Fn(&Graph, &Partition) -> ScoreValue + Send + Sync>,
    ) -> Partition;
}

mod short_bursts;
mod tempering;
mod verbose_bursts;
pub use short_bursts::ShortBurstsOptimizer;
pub use tempering::ParallelTemperingOptimizer;
pub use verbose_bursts::VerboseBurstsOptimizer;
