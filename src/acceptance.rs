use crate::graph::Graph;
use crate::partition::Partition;
use crate::gingleator::ScoreValue;
// use std::cmp::{max, min};

pub type AcceptanceProb = f64;

pub fn mod_hill_climbing(
    obj_fn: impl Fn(&Graph, &Partition) -> ScoreValue + Send + Clone + Copy,
    graph: &Graph,
    proposed_partition: &Partition,
    parent_partition: &Partition,
) -> AcceptanceProb {
    let c = 20000.0;
    let proposed_score = obj_fn(&graph, &proposed_partition);
    let parent_score = obj_fn(&graph, &parent_partition);
    if proposed_score >= parent_score {
        1.0
    }
    else {
        let prob = (c * (proposed_score - parent_score)).exp();
        prob.min(1.0)
    }
}


pub fn dallas_birmingham_together(
    _graph: &Graph,
    proposed_partition: &Partition,
    _parent_partition: &Partition,
) -> AcceptanceProb {
// id 1301 and 626 im same district
    (proposed_partition.assignments[626] == proposed_partition.assignments[1301]) as i64 as f64
}

// pub fn dallas_accept_fn(
//     graph: &Graph,
//     partition: &Partition,
// ) -> AcceptanceProb {
//     mod_hill_climbing(&graph, &partition) * dallas_birmingham_together(&graph, &partition)
// }

pub fn mod_hill_climbing_accept_fn(
    opt_fun: impl Fn(&Graph, &Partition) -> ScoreValue + Send + Clone + Copy,
) -> impl Fn(&Graph, &Partition, &Partition) -> AcceptanceProb + Send + Clone + Copy {
    move |graph: &Graph, proposed_partition: &Partition, parent_partition: &Partition| -> AcceptanceProb {
        mod_hill_climbing(opt_fun, &graph, &proposed_partition, &parent_partition)
    }
}