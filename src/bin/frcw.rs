//! Main CLI for frcw.
use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use clap::{value_t, App, Arg};
use frcw::init::from_networkx;
use frcw::recom::run::multi_chain;
use frcw::recom::{RecomParams, RecomVariant};
use frcw::stats::{JSONLWriter, StatsWriter, TSVWriter};
use serde_json::json;
use sha3::{Digest, Sha3_256};
use std::path::PathBuf;
use std::{fs, io};

fn main() {
    let matches = App::new("frcw")
        .version("0.1.0")
        .author("Parker J. Rule <parker.rule@tufts.edu>")
        .about("A minimal implementation of the ReCom Markov chain")
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
            Arg::with_name("balance_ub")
                .long("balance-ub")
                .short("M") // Variable used in RevReCom paper
                .takes_value(true)
                .default_value("0") // TODO: just use unwrap_or_default() instead?
                .help("The normalizing constant (reversible ReCom only)."),
        )
        .arg(
            Arg::with_name("n_threads")
                .long("n-threads")
                .takes_value(true)
                .required(true)
                .help("The number of threads to use."),
        )
        .arg(
            Arg::with_name("batch_size")
                .long("batch-size")
                .takes_value(true)
                .required(true)
                .help("The number of proposals per batch job."),
        )
        .arg(
            Arg::with_name("variant")
                .long("variant")
                .takes_value(true)
                .default_value("reversible"),
        ) // other options: cut_edges, district_pairs
        .arg(
            Arg::with_name("writer")
                .long("writer")
                .takes_value(true)
                .default_value("jsonl"),
        ) // other options: jsonl-full, tsv
        .arg(
            Arg::with_name("sum_cols")
                .long("sum-cols")
                .multiple(true)
                .takes_value(true),
        )
        .arg(Arg::with_name("spanning_tree_counts").long("st-counts"))
        .get_matches();
    let n_steps = value_t!(matches.value_of("n_steps"), u64).unwrap_or_else(|e| e.exit());
    let rng_seed = value_t!(matches.value_of("rng_seed"), u64).unwrap_or_else(|e| e.exit());
    let tol = value_t!(matches.value_of("tol"), f64).unwrap_or_else(|e| e.exit());
    let balance_ub = value_t!(matches.value_of("balance_ub"), u32).unwrap_or_else(|e| e.exit());
    let n_threads = value_t!(matches.value_of("n_threads"), usize).unwrap_or_else(|e| e.exit());
    let batch_size = value_t!(matches.value_of("batch_size"), usize).unwrap_or_else(|e| e.exit());
    let graph_json = fs::canonicalize(PathBuf::from(matches.value_of("graph_json").unwrap()))
        .unwrap()
        .into_os_string()
        .into_string()
        .unwrap();
    let pop_col = matches.value_of("pop_col").unwrap();
    let assignment_col = matches.value_of("assignment_col").unwrap();
    let variant_str = matches.value_of("variant").unwrap();
    let writer_str = matches.value_of("writer").unwrap();
    let st_counts = matches.is_present("spanning_tree_counts");
    let sum_cols = matches
        .values_of("sum_cols")
        .unwrap_or_default()
        .map(|c| c.to_string())
        .collect();

    let variant = match variant_str {
        "reversible" => RecomVariant::Reversible,
        "cut_edges" => RecomVariant::CutEdges,
        "district_pairs" => RecomVariant::DistrictPairs,
        bad => panic!("Parameter error: invalid variant '{}'", bad),
    };
    let writer: Box<dyn StatsWriter> = match writer_str {
        "tsv" => Box::new(TSVWriter::new()),
        "jsonl" => Box::new(JSONLWriter::new(false, st_counts)),
        "jsonl-full" => Box::new(JSONLWriter::new(true, st_counts)),
        bad => panic!("Parameter error: invalid writer '{}'", bad),
    };
    if variant == RecomVariant::Reversible && balance_ub == 0 {
        panic!("For reversible ReCom, specify M > 0.");
    }

    assert!(tol >= 0.0 && tol <= 1.0);

    let (graph, partition) = from_networkx(&graph_json, pop_col, assignment_col, sum_cols).unwrap();
    let avg_pop = (graph.total_pop as f64) / (partition.num_dists as f64);
    let params = RecomParams {
        min_pop: ((1.0 - tol) * avg_pop as f64).floor() as u32,
        max_pop: ((1.0 + tol) * avg_pop as f64).ceil() as u32,
        num_steps: n_steps,
        rng_seed: rng_seed,
        balance_ub: balance_ub,
        variant: variant,
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
        "batch_size": batch_size,
        "rng_seed": rng_seed,
        "num_threads": n_threads,
        "num_steps": n_steps,
        "parallel": true,
        "graph_json": graph_json
    });
    if variant == RecomVariant::Reversible {
        meta.as_object_mut()
            .unwrap()
            .insert("balance_ub".to_string(), json!(balance_ub));
    }
    println!("{}", json!({ "meta": meta }).to_string());
    multi_chain(&graph, &partition, writer, params, n_threads, batch_size);
}
