use crate::graph::Graph;
use crate::partition::Partition;

pub fn mod_hill_climbing(
    graph: &Graph,
    partition: &Partition,
) -> f64 {
    0.0
}


pub fn dallas_birmingham_together(
    graph: &Graph,
    partition: &Partition,
) -> f64 {
// id 1301 and 626 im same district
    (partition.assignments[626] == partition.assignments[1301]) as i64 as f64
}

pub fn dallas_accept_fn(
    graph: &Graph,
    partition: &Partition,
) -> f64 {
    mod_hill_climbing(&graph, &partition) * dallas_birmingham_together(&graph, &partition)
}