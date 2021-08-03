//! Reversible ReCom distribution test for the frcw engine.
//!
//! Intended for use with the MGGG benchmark suite
//! (https://github.com/mggg/benchmarks).
use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use clap::{value_t, App, Arg};
use frcw::graph::Graph;
use frcw::partition::Partition;
use frcw::recom::run::multi_chain;
use frcw::recom::{RecomParams, RecomVariant};
use frcw::stats::{StatsWriter, AssignmentsOnlyWriter};
use std::fs::read_to_string;

fn main() {
    let matches = App::new("frcw-revrecom-dist-test")
        .version("0.1.0")
        .author("Parker J. Rule <parker.rule@tufts.edu>")
        .about("RevReCom distribution tests for frcw")
        .arg(
            Arg::with_name("graph_file")
                .long("graph-file")
                .takes_value(true)
                .required(true)
                .help("The path of the dual graph (in edge list format).")
        )
        .arg(
            Arg::with_name("pop_file")
            .long("pop-file")
            .takes_value(true)
            .required(true)
            .help("The path of the population file for the dual graph.")
        )
        .arg(
            Arg::with_name("assignment_file")
            .long("assignment-file")
            .takes_value(true)
            .required(true)
            .help("The path of the seed plan assignment for the dual graph.")
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
        ).get_matches();
       
    let n_steps = value_t!(matches.value_of("n_steps"), u64).unwrap_or_else(|e| e.exit());
    let rng_seed = value_t!(matches.value_of("rng_seed"), u64).unwrap_or_else(|e| e.exit());
    let tol = value_t!(matches.value_of("tol"), f64).unwrap_or_else(|e| e.exit());
    let balance_ub = value_t!(matches.value_of("balance_ub"), u32).unwrap_or_else(|e| e.exit());
    let n_threads = value_t!(matches.value_of("n_threads"), usize).unwrap_or_else(|e| e.exit());
    let batch_size = value_t!(matches.value_of("batch_size"), usize).unwrap_or_else(|e| e.exit());
    let graph_path = matches.value_of("graph_file").unwrap();
    let pop_path = matches.value_of("pop_file").unwrap();
    let assignments_path = matches.value_of("assignment_file").unwrap();

    assert!(tol >= 0.0 && tol <= 1.0);
    assert!(balance_ub > 0);

    let graph_data = read_to_string(graph_path).expect("Could not read edge list file");
    let pop_data = read_to_string(pop_path).expect("Could not read population file");
    let assignments_data = read_to_string(assignments_path).expect(
        "Could not read assignment file");

    let graph = Graph::from_edge_list(&graph_data, &pop_data).unwrap();
    let partition = Partition::from_assignment_str(&graph, &assignments_data).unwrap();
    let avg_pop = (graph.total_pop as f64) / (partition.num_dists as f64);
    let params = RecomParams {
        min_pop: ((1.0 - tol) * avg_pop as f64).floor() as u32,
        max_pop: ((1.0 + tol) * avg_pop as f64).ceil() as u32,
        num_steps: n_steps,
        rng_seed: rng_seed,
        balance_ub: balance_ub, 
        variant: RecomVariant::Reversible,
    };
    let writer: Box<dyn StatsWriter> = Box::new(AssignmentsOnlyWriter::new());
    multi_chain(&graph, &partition, writer, params, n_threads, batch_size);
}
