//! Short bursts optimization CLI for frcw.
use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use clap::{value_t, App, Arg};
use frcw::config::parse_region_weights_config;
use frcw::graph::Graph;
use frcw::init::from_networkx;
use frcw::partition::Partition;
use frcw::recom::opt::{multi_short_bursts, ScoreValue};
use frcw::recom::{RecomParams, RecomVariant};
use frcw::stats::partition_attr_sums;
use frcw::acceptance::dallas_birmingham_together;
use frcw::gingleator::make_objective_fn;
use serde_json::json;
use serde_json::Value;
use sha3::{Digest, Sha3_256};
use std::marker::Send;
use std::path::PathBuf;
use std::{fs, io};

fn main() {
    let cli = App::new("frcw_short_bursts")
        .version("0.1.0")
        .author("Parker J. Rule <parker.rule@tufts.edu>")
        .about("A short bursts optimizer for redistricting")
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
                .required(true)
                .help("The number of threads to use."),
        )
        .arg(
            Arg::with_name("burst_length")
                .long("burst-length")
                .takes_value(true)
                .required(true)
                .help("The number of accepted steps per short burst."),
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
            Arg::with_name("accept_fn")
                .long("accept-fn")
                .takes_value(true)
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
    let objective_config = matches.value_of("objective").unwrap();
    let objective_fn = make_objective_fn(objective_config);
    let region_weights_raw = matches.value_of("region_weights").unwrap_or_default();
    let region_weights = parse_region_weights_config(region_weights_raw);
    let accept_fn_name = matches.value_of("accept_fn").unwrap_or_default();
    let accept_fn = match accept_fn_name {
        "" => None,
        "dallas" => Some(dallas_birmingham_together),
        _ => panic!("Unknown acceptance function '{}'.", accept_fn_name)
    };

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
    multi_short_bursts(
        &graph,
        partition,
        &params,
        n_threads,
        objective_fn,
        accept_fn,
        burst_length,
        true,
    );
}
