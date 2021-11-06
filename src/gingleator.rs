use crate::graph::Graph;
use crate::partition::Partition;
use crate::stats::partition_attr_sums;
use serde_json::Value;
pub type ScoreValue = f64;

pub fn make_objective_fn(
    objective_config: &str,
) -> impl Fn(&Graph, &Partition) -> ScoreValue + Send + Clone + Copy {
    let data: Value = serde_json::from_str(objective_config).unwrap();
    let obj_type = data["objective"].as_str().unwrap();

    // For now, we only support Gingles optimization with next-partial-district
    // augmentation. see https://github.com/vrdi/shortbursts-gingles/blob/
    // d9fb26ec313cd93ac80171b23095c4f3dfab0422/state_experiments/gingleator.py#L253
    assert!(obj_type == "gingles_partial");
    let threshold = data["threshold"].as_f64().unwrap();
    assert!(threshold > 0.0 && threshold < 1.0);

    // A hack to share strings between threads:
    // https://stackoverflow.com/a/52367953
    let min_pop_col = &*Box::leak(
        data["min_pop"]
            .as_str()
            .unwrap()
            .to_owned()
            .into_boxed_str(),
    );
    let total_pop_col = &*Box::leak(
        data["total_pop"]
            .as_str()
            .unwrap()
            .to_owned()
            .into_boxed_str(),
    );

    move |graph: &Graph, partition: &Partition| -> ScoreValue {
        let min_pops = partition_attr_sums(graph, partition, min_pop_col);
        let total_pops = partition_attr_sums(graph, partition, total_pop_col);
        let shares: Vec<f64> = min_pops
            .iter()
            .zip(total_pops.iter())
            .map(|(&m, &t)| m as f64 / t as f64)
            .collect();
        let opportunity_count = shares.iter().filter(|&s| s >= &threshold).count();

        // partial ordering on f64:
        // see https://www.reddit.com/r/rust/comments/29kia3/no_ord_for_f32/
        // see https://doc.rust-lang.org/std/vec/struct.Vec.html#method.sort_by
        let mut sorted_below = shares
            .into_iter()
            .filter(|s| s < &threshold)
            .collect::<Vec<f64>>();
        sorted_below.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let next_highest = match sorted_below.last() {
            Some(&v) => v,
            None => 0.0,
        };
        opportunity_count as f64 + next_highest
    }
}