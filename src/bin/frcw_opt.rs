//! Optimization CLI for frcw.
use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use clap::{value_t, App, Arg};
use frcw::config::parse_region_weights_config;
use frcw::graph::Graph;
use frcw::init::from_networkx;
use frcw::partition::Partition;
use frcw::recom::opt::{Optimizer, ParallelTemperingOptimizer, ScoreValue, ShortBurstsOptimizer};
use frcw::recom::{RecomParams, RecomVariant};
use frcw::stats::partition_attr_sums;
use serde_json::json;
use serde_json::Value;
use sha3::{Digest, Sha3_256};
use std::marker::Send;
use std::path::PathBuf;
use std::{fs, io};

fn make_gingles_partial(
    objective_config: &str,
) -> impl Fn(&Graph, &Partition) -> ScoreValue + Send + Copy {
    let data: Value = serde_json::from_str(objective_config).unwrap();
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
        opportunity_count as f64 + (next_highest / threshold)
    }
}

fn make_gingles_coalition(
    objective_config: &str,
) -> impl Fn(&Graph, &Partition) -> ScoreValue + Send + Copy {
    let data: Value = serde_json::from_str(objective_config).unwrap();
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
    let coalition_pop_col = &*Box::leak(
        data["coalition_pop"]
            .as_str()
            .unwrap()
            .to_owned()
            .into_boxed_str(),
    );

    move |graph: &Graph, partition: &Partition| -> ScoreValue {
        let min_pops = partition_attr_sums(graph, partition, min_pop_col);
        let total_pops = partition_attr_sums(graph, partition, total_pop_col);
        let coalition_pops = partition_attr_sums(graph, partition, coalition_pop_col);
        let min_shares: Vec<f64> = min_pops
            .iter()
            .zip(total_pops.iter())
            .map(|(&m, &t)| m as f64 / t as f64)
            .collect();
        let coalition_shares: Vec<f64> = coalition_pops
            .iter()
            .zip(total_pops.iter())
            .map(|(&c, &t)| c as f64 / t as f64)
            .collect();

        let min_count = min_shares.iter().filter(|&s| s >= &threshold).count();
        let coalition_no_min_count = min_shares
            .iter()
            .zip(coalition_shares.iter())
            .filter(|(&ms, &cs)| ms < threshold && cs >= threshold)
            .count();

        // partial ordering on f64:
        // see https://www.reddit.com/r/rust/comments/29kia3/no_ord_for_f32/
        // see https://doc.rust-lang.org/std/vec/struct.Vec.html#method.sort_by
        let mut min_sorted_below = min_shares
            .into_iter()
            .filter(|s| s < &threshold)
            .collect::<Vec<f64>>();
        min_sorted_below.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let next_highest_min = match min_sorted_below.last() {
            Some(&v) => v,
            None => 0.0,
        };

        let mut coalition_sorted_below = coalition_shares
            .into_iter()
            .filter(|s| s < &threshold)
            .collect::<Vec<f64>>();
        coalition_sorted_below.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let next_highest_coalition = match coalition_sorted_below.last() {
            Some(&v) => v,
            None => 0.0,
        };

        min_count as f64
            + (0.75 * coalition_no_min_count as f64)
            + next_highest_min
            + next_highest_coalition
    }
}

fn make_effectiveness(
    objective_config: &str,
) -> impl Fn(&Graph, &Partition) -> ScoreValue + Send + Copy {
    let data: Value = serde_json::from_str(objective_config).unwrap();

    let elections = data["elections"].as_object().unwrap();
    let primaries = elections["primaries"].as_array().unwrap();
    let generals = elections["generals"].as_array().unwrap();
    let primary_cutoff = elections["primary_cutoff"].as_u64().unwrap() as usize;
    let general_cutoff = elections["general_cutoff"].as_u64().unwrap() as usize;

    let primary_pairs: Vec<(String, String)> = primaries
        .iter()
        .map(|primary| {
            let pref_col = primary["preferred"].as_str().unwrap().to_owned();
            let other_col = primary["other"].as_str().unwrap().to_owned();
            (pref_col, other_col)
        })
        .collect();
    let general_pairs: Vec<(String, String)> = generals
        .iter()
        .map(|general| {
            let pref_col = general["preferred"].as_str().unwrap().to_owned();
            let other_col = general["other"].as_str().unwrap().to_owned();
            (pref_col, other_col)
        })
        .collect();

    let primary_pairs_sh = &*Box::leak(Box::new(primary_pairs));
    let general_pairs_sh = &*Box::leak(Box::new(general_pairs));
    let num_primaries = primary_pairs_sh.len() as f64;
    let num_generals = general_pairs_sh.len() as f64;

    move |graph: &Graph, partition: &Partition| -> ScoreValue {
        let mut primary_counts = vec![0; partition.num_dists as usize];
        let mut general_counts = vec![0; partition.num_dists as usize];

        for (pref_col, other_col) in primary_pairs_sh.iter() {
            let pref_sums = partition_attr_sums(graph, partition, pref_col);
            let other_sums = partition_attr_sums(graph, partition, other_col);
            for (dist, (p, o)) in pref_sums.iter().zip(other_sums.iter()).enumerate() {
                if p >= o {
                    primary_counts[dist] += 1;
                }
            }
        }
        for (pref_col, other_col) in general_pairs_sh.iter() {
            let pref_sums = partition_attr_sums(graph, partition, pref_col);
            let other_sums = partition_attr_sums(graph, partition, other_col);
            for (dist, (p, o)) in pref_sums.iter().zip(other_sums.iter()).enumerate() {
                if p >= o {
                    general_counts[dist] += 1;
                }
            }
        }

        let cutoff_count = primary_counts
            .iter()
            .zip(general_counts.iter())
            .filter(|(pc, gc)| **pc >= primary_cutoff && **gc >= general_cutoff)
            .count() as f64;
        let mut partials: Vec<f64> = primary_counts
            .iter()
            .zip(general_counts.iter())
            .filter(|(pc, gc)| **pc < primary_cutoff || **gc < general_cutoff)
            .map(|(pc, gc)| (*pc as f64 / num_primaries) * (*gc as f64 / num_generals))
            .collect();
        partials.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let highest_partial = match partials.last() {
            Some(&v) => v,
            None => 0.0,
        };
        cutoff_count + highest_partial
    }
}

fn make_effectiveness_gingles(
    objective_config: &str,
) -> impl Fn(&Graph, &Partition) -> ScoreValue + Send + Copy {
    let data: Value = serde_json::from_str(objective_config).unwrap();

    let elections = data["elections"].as_object().unwrap();
    let primaries = elections["primaries"].as_array().unwrap();
    let generals = elections["generals"].as_array().unwrap();
    let primary_cutoff = elections["primary_cutoff"].as_u64().unwrap() as usize;
    let general_cutoff = elections["general_cutoff"].as_u64().unwrap() as usize;

    let primary_pairs: Vec<(String, String)> = primaries
        .iter()
        .map(|primary| {
            let pref_col = primary["preferred"].as_str().unwrap().to_owned();
            let other_col = primary["other"].as_str().unwrap().to_owned();
            (pref_col, other_col)
        })
        .collect();
    let general_pairs: Vec<(String, String)> = generals
        .iter()
        .map(|general| {
            let pref_col = general["preferred"].as_str().unwrap().to_owned();
            let other_col = general["other"].as_str().unwrap().to_owned();
            (pref_col, other_col)
        })
        .collect();

    let primary_pairs_sh = &*Box::leak(Box::new(primary_pairs));
    let general_pairs_sh = &*Box::leak(Box::new(general_pairs));
    let num_primaries = primary_pairs_sh.len() as f64;
    let num_generals = general_pairs_sh.len() as f64;

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
        let mut primary_counts = vec![0; partition.num_dists as usize];
        let mut general_counts = vec![0; partition.num_dists as usize];

        for (pref_col, other_col) in primary_pairs_sh.iter() {
            let pref_sums = partition_attr_sums(graph, partition, pref_col);
            let other_sums = partition_attr_sums(graph, partition, other_col);
            for (dist, (p, o)) in pref_sums.iter().zip(other_sums.iter()).enumerate() {
                if p >= o {
                    primary_counts[dist] += 1;
                }
            }
        }
        for (pref_col, other_col) in general_pairs_sh.iter() {
            let pref_sums = partition_attr_sums(graph, partition, pref_col);
            let other_sums = partition_attr_sums(graph, partition, other_col);
            for (dist, (p, o)) in pref_sums.iter().zip(other_sums.iter()).enumerate() {
                if p >= o {
                    general_counts[dist] += 1;
                }
            }
        }

        let cutoff_count = primary_counts
            .iter()
            .zip(general_counts.iter())
            .filter(|(pc, gc)| **pc >= primary_cutoff && **gc >= general_cutoff)
            .count() as f64;
        let mut partials: Vec<f64> = primary_counts
            .iter()
            .zip(general_counts.iter())
            .filter(|(pc, gc)| **pc < primary_cutoff || **gc < general_cutoff)
            .map(|(pc, gc)| (*pc as f64 / num_primaries) * (*gc as f64 / num_generals))
            .collect();
        partials.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let highest_partial = match partials.last() {
            Some(&v) => v,
            None => 0.0,
        };
        let eff_score = cutoff_count + highest_partial;

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
        let gingles_score = opportunity_count as f64 + (next_highest / threshold);
        eff_score + gingles_score
    }
}


fn make_effectiveness_gingles_two_way(
    objective_config: &str,
) -> impl Fn(&Graph, &Partition) -> ScoreValue + Send + Copy {
    let data: Value = serde_json::from_str(objective_config).unwrap();

    let elections = data["elections"].as_object().unwrap();
    let primaries = elections["primaries"].as_array().unwrap();
    let generals = elections["generals"].as_array().unwrap();
    let primary_cutoff = elections["primary_cutoff"].as_u64().unwrap() as usize;
    let general_cutoff = elections["general_cutoff"].as_u64().unwrap() as usize;

    let primary_pairs: Vec<(String, String)> = primaries
        .iter()
        .map(|primary| {
            let pref_col = primary["preferred"].as_str().unwrap().to_owned();
            let other_col = primary["other"].as_str().unwrap().to_owned();
            (pref_col, other_col)
        })
        .collect();
    let general_pairs: Vec<(String, String)> = generals
        .iter()
        .map(|general| {
            let pref_col = general["preferred"].as_str().unwrap().to_owned();
            let other_col = general["other"].as_str().unwrap().to_owned();
            (pref_col, other_col)
        })
        .collect();

    let primary_pairs_sh = &*Box::leak(Box::new(primary_pairs));
    let general_pairs_sh = &*Box::leak(Box::new(general_pairs));
    let num_primaries = primary_pairs_sh.len() as f64;
    let num_generals = general_pairs_sh.len() as f64;

    let threshold = data["threshold"].as_f64().unwrap();
    assert!(threshold > 0.0 && threshold < 1.0);

    let min_pop1_col = &*Box::leak(
        data["min_pop1"]
            .as_str()
            .unwrap()
            .to_owned()
            .into_boxed_str(),
    );
    let total_pop1_col = &*Box::leak(
        data["total_pop1"]
            .as_str()
            .unwrap()
            .to_owned()
            .into_boxed_str(),
    );
    let min_pop2_col = &*Box::leak(
        data["min_pop2"]
            .as_str()
            .unwrap()
            .to_owned()
            .into_boxed_str(),
    );
    let total_pop2_col = &*Box::leak(
        data["total_pop2"]
            .as_str()
            .unwrap()
            .to_owned()
            .into_boxed_str(),
    );


    move |graph: &Graph, partition: &Partition| -> ScoreValue {
        let mut primary_counts = vec![0; partition.num_dists as usize];
        let mut general_counts = vec![0; partition.num_dists as usize];

        for (pref_col, other_col) in primary_pairs_sh.iter() {
            let pref_sums = partition_attr_sums(graph, partition, pref_col);
            let other_sums = partition_attr_sums(graph, partition, other_col);
            for (dist, (p, o)) in pref_sums.iter().zip(other_sums.iter()).enumerate() {
                if p >= o {
                    primary_counts[dist] += 1;
                }
            }
        }
        for (pref_col, other_col) in general_pairs_sh.iter() {
            let pref_sums = partition_attr_sums(graph, partition, pref_col);
            let other_sums = partition_attr_sums(graph, partition, other_col);
            for (dist, (p, o)) in pref_sums.iter().zip(other_sums.iter()).enumerate() {
                if p >= o {
                    general_counts[dist] += 1;
                }
            }
        }

        let cutoff_count = primary_counts
            .iter()
            .zip(general_counts.iter())
            .filter(|(pc, gc)| **pc >= primary_cutoff && **gc >= general_cutoff)
            .count() as f64;
        let mut partials: Vec<f64> = primary_counts
            .iter()
            .zip(general_counts.iter())
            .filter(|(pc, gc)| **pc < primary_cutoff || **gc < general_cutoff)
            .map(|(pc, gc)| (*pc as f64 / num_primaries) * (*gc as f64 / num_generals))
            .collect();
        partials.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let highest_partial = match partials.last() {
            Some(&v) => v,
            None => 0.0,
        };
        let eff_score = cutoff_count + highest_partial;

        let min_pops1 = partition_attr_sums(graph, partition, min_pop1_col);
        let total_pops1 = partition_attr_sums(graph, partition, total_pop1_col);
        let shares: Vec<f64> = min_pops1
            .iter()
            .zip(total_pops1.iter())
            .map(|(&m, &t)| m as f64 / t as f64)
            .collect();
        let opportunity_count1 = shares.iter().filter(|&s| s >= &threshold).count();

        // partial ordering on f64:
        // see https://www.reddit.com/r/rust/comments/29kia3/no_ord_for_f32/
        // see https://doc.rust-lang.org/std/vec/struct.Vec.html#method.sort_by
        let mut sorted_below1 = shares
            .into_iter()
            .filter(|s| s < &threshold)
            .collect::<Vec<f64>>();
        sorted_below1.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let next_highest1 = match sorted_below1.last() {
            Some(&v) => v,
            None => 0.0,
        };
        let gingles1_score = opportunity_count1 as f64 + (next_highest1 / threshold);

        let min_pops2 = partition_attr_sums(graph, partition, min_pop2_col);
        let total_pops2 = partition_attr_sums(graph, partition, total_pop2_col);
        let shares: Vec<f64> = min_pops2
            .iter()
            .zip(total_pops2.iter())
            .map(|(&m, &t)| m as f64 / t as f64)
            .collect();
        let opportunity_count2 = shares.iter().filter(|&s| s >= &threshold).count();

        let mut sorted_below2 = shares
            .into_iter()
            .filter(|s| s < &threshold)
            .collect::<Vec<f64>>();
        sorted_below2.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let next_highest2 = match sorted_below2.last() {
            Some(&v) => v,
            None => 0.0,
        };
        let gingles2_score = opportunity_count2 as f64 + (next_highest2 / threshold);

        eff_score + gingles1_score + gingles2_score
    }
}

fn make_effectiveness_gingles_intersection(
    objective_config: &str,
) -> impl Fn(&Graph, &Partition) -> ScoreValue + Send + Copy {
    let data: Value = serde_json::from_str(objective_config).unwrap();

    let elections = data["elections"].as_object().unwrap();
    let primaries = elections["primaries"].as_array().unwrap();
    let generals = elections["generals"].as_array().unwrap();
    let primary_cutoff = elections["primary_cutoff"].as_u64().unwrap() as usize;
    let general_cutoff = elections["general_cutoff"].as_u64().unwrap() as usize;

    let primary_pairs: Vec<(String, String)> = primaries
        .iter()
        .map(|primary| {
            let pref_col = primary["preferred"].as_str().unwrap().to_owned();
            let other_col = primary["other"].as_str().unwrap().to_owned();
            (pref_col, other_col)
        })
        .collect();
    let general_pairs: Vec<(String, String)> = generals
        .iter()
        .map(|general| {
            let pref_col = general["preferred"].as_str().unwrap().to_owned();
            let other_col = general["other"].as_str().unwrap().to_owned();
            (pref_col, other_col)
        })
        .collect();

    let primary_pairs_sh = &*Box::leak(Box::new(primary_pairs));
    let general_pairs_sh = &*Box::leak(Box::new(general_pairs));
    let num_primaries = primary_pairs_sh.len() as f64;
    let num_generals = general_pairs_sh.len() as f64;

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
        let mut primary_counts = vec![0; partition.num_dists as usize];
        let mut general_counts = vec![0; partition.num_dists as usize];

        for (pref_col, other_col) in primary_pairs_sh.iter() {
            let pref_sums = partition_attr_sums(graph, partition, pref_col);
            let other_sums = partition_attr_sums(graph, partition, other_col);
            for (dist, (p, o)) in pref_sums.iter().zip(other_sums.iter()).enumerate() {
                if p >= o {
                    primary_counts[dist] += 1;
                }
            }
        }
        for (pref_col, other_col) in general_pairs_sh.iter() {
            let pref_sums = partition_attr_sums(graph, partition, pref_col);
            let other_sums = partition_attr_sums(graph, partition, other_col);
            for (dist, (p, o)) in pref_sums.iter().zip(other_sums.iter()).enumerate() {
                if p >= o {
                    general_counts[dist] += 1;
                }
            }
        }
        let min_pops = partition_attr_sums(graph, partition, min_pop_col);
        let total_pops = partition_attr_sums(graph, partition, total_pop_col);
        let min_shares: Vec<f64> = min_pops
            .iter()
            .zip(total_pops.iter())
            .map(|(&m, &t)| m as f64 / t as f64)
            .collect();

        let all_count = primary_counts
            .iter()
            .zip(general_counts.iter())
            .zip(min_shares.iter())
            .filter(|((pc, gc), ms)| **pc >= primary_cutoff && **gc >= general_cutoff && **ms >= threshold)
            .count() as f64;

        let mut min_below = min_shares
            .into_iter()
            .filter(|s| s < &threshold)
            .collect::<Vec<f64>>();
        min_below.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let next_highest = match min_below.last() {
            Some(&v) => v,
            None => 0.0,
        };
        all_count + next_highest
    }
}

fn main() {
    let cli = App::new("frcw_opt")
        .version("0.1.1")
        .author("Parker J. Rule <parker.rule@tufts.edu>")
        .about("An  optimizer for redistricting")
        .arg(
            Arg::with_name("graph_json")
                .long("graph-json")
                .takes_value(true)
                .required(true)
                .help("The path of the dual graph (in NetworkX format)."),
        )
        .arg(
            Arg::with_name("n_steps")
                .long("n-steps")
                .takes_value(true)
                .required(true)
                .help("The number of proposals to generate."),
        )
        .arg(
            Arg::with_name("tol")
                .long("tol")
                .takes_value(true)
                .required(true)
                .help("The relative population tolerance."),
        )
        .arg(
            Arg::with_name("pop_col")
                .long("pop-col")
                .takes_value(true)
                .required(true)
                .help("The name of the total population column in the graph metadata."),
        )
        .arg(
            Arg::with_name("assignment_col")
                .long("assignment-col")
                .takes_value(true)
                .required(true)
                .help("The name of the assignment column in the graph metadata."),
        )
        .arg(
            Arg::with_name("rng_seed")
                .long("rng-seed")
                .takes_value(true)
                .required(true)
                .help("The seed of the RNG used to draw proposals."),
        )
        .arg(
            Arg::with_name("n_threads")
                .long("n-threads")
                .takes_value(true)
                .help("The number of threads to use (short bursts mode)."),
        )
        .arg(
            Arg::with_name("temperatures")
                .long("temperatures")
                .takes_value(true)
                .multiple(true)
                .help("The temperatures to use (tempering mode)."),
        )
        .arg(
            Arg::with_name("burst_length")
                .long("burst-length")
                .takes_value(true)
                .required(true)
                .help("The number of accepted steps per batch (short bursts/tempering)."),
        )
        .arg(
            Arg::with_name("sum_cols")
                .long("sum-cols")
                .multiple(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("objective")
                .long("objective")
                .takes_value(true)
                .required(true)
                .help("A JSON-formatted objective function configuration>"),
        )
        .arg(
            Arg::with_name("region_weights")
                .long("region-weights")
                .takes_value(true)
                .help("Region columns with weights for region-aware ReCom."),
        )
        .arg(
            Arg::with_name("optimizer")
                .long("optimizer")
                .takes_value(true)
                .default_value("short_bursts")
                .help("The optimizer to use (options: short_bursts, tempering)."),
        );
    let matches = cli.get_matches();
    let n_steps = value_t!(matches.value_of("n_steps"), u64).unwrap_or_else(|e| e.exit());
    let n_threads = value_t!(matches.value_of("n_threads"), usize).unwrap_or_else(|e| e.exit());
    let rng_seed = value_t!(matches.value_of("rng_seed"), u64).unwrap_or_else(|e| e.exit());
    let tol = value_t!(matches.value_of("tol"), f64).unwrap_or_else(|e| e.exit());
    let burst_length =
        value_t!(matches.value_of("burst_length"), usize).unwrap_or_else(|e| e.exit());
    let graph_json = fs::canonicalize(PathBuf::from(matches.value_of("graph_json").unwrap()))
        .unwrap()
        .into_os_string()
        .into_string()
        .unwrap();
    let pop_col = matches.value_of("pop_col").unwrap();
    let assignment_col = matches.value_of("assignment_col").unwrap();
    let sum_cols = matches
        .values_of("sum_cols")
        .unwrap_or_default()
        .map(|c| c.to_string())
        .collect();

    let objective_config = matches
        .value_of("objective")
        .unwrap()
        .replace("\\{", "{")
        .replace("\\}", "}")
        .replace("\\\"", "\"");
    let obj_data: Value = serde_json::from_str(&objective_config).unwrap();
    let obj_type = obj_data["objective"].as_str().unwrap();
    //let objective_fn = make_effectiveness_gingles_two_way(&objective_config);
    let objective_fn = make_gingles_partial(&objective_config);
    //let objective_fn = make_effectiveness_gingles_intersection(&objective_config);
    let region_weights_raw = matches.value_of("region_weights").unwrap_or_default();
    let region_weights = parse_region_weights_config(region_weights_raw);
    let optimizer_name = matches.value_of("optimizer").unwrap();
    let temperatures: Vec<f64> = matches
        .values_of("temperatures")
        .unwrap_or_default()
        .map(|c| c.parse::<f64>().unwrap())
        .collect();

    assert!(tol >= 0.0 && tol <= 1.0);

    let (graph, partition) = from_networkx(&graph_json, pop_col, assignment_col, sum_cols).unwrap();
    let avg_pop = (graph.total_pop as f64) / (partition.num_dists as f64);
    let params = RecomParams {
        min_pop: ((1.0 - tol) * avg_pop as f64).floor() as u32,
        max_pop: ((1.0 + tol) * avg_pop as f64).ceil() as u32,
        num_steps: n_steps,
        rng_seed: rng_seed,
        balance_ub: 0,
        variant: match region_weights {
            None => RecomVariant::DistrictPairsRMST,
            Some(_) => RecomVariant::DistrictPairsRegionAware,
        },
        region_weights: region_weights.clone(),
    };

    let mut graph_file = fs::File::open(&graph_json).unwrap();
    let mut graph_hasher = Sha3_256::new();
    io::copy(&mut graph_file, &mut graph_hasher).unwrap();
    let graph_hash = format!("{:x}", graph_hasher.finalize());
    let mut meta = json!({
        "assignment_col": assignment_col,
        "tol": tol,
        "pop_col": pop_col,
        "graph_path": graph_json,
        "graph_sha3": graph_hash,
        "rng_seed": rng_seed,
        "num_threads": n_threads,
        "num_steps": n_steps,
        "parallel": true,
        "type": "short_bursts",
        "burst_length": burst_length,
        "graph_json": graph_json,
    });
    if region_weights.is_some() {
        meta.as_object_mut()
            .unwrap()
            .insert("region_weights".to_string(), json!(region_weights));
    }
    println!("{}", json!({ "meta": meta }).to_string());

    if optimizer_name == "short_bursts" {
        ShortBurstsOptimizer::new(params, n_threads, burst_length, true).optimize(
            &graph,
            partition,
            objective_fn,
        );
    } else if optimizer_name == "tempering" {
        assert!(temperatures.len() > 0);
        ParallelTemperingOptimizer::new(params, temperatures, burst_length, true).optimize(
            &graph,
            partition,
            objective_fn,
        );
    } else {
        panic!("Unknown optimizer.");
    }
}
