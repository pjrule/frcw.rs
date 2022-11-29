//! CLI for spanning tree count processing.
use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use clap::{value_t, App, Arg};
use frcw::config::parse_region_weights_config;
use frcw::init::graph_from_networkx;
use frcw::recom::run::multi_chain;
use frcw::recom::{RecomParams, RecomVariant};
use frcw::stats::{JSONLWriter, PcompressWriter, StatsWriter, TSVWriter};
use serde_json::json;
use sha3::{Digest, Sha3_256};
use std::path::PathBuf;
use std::{fs, io};

fn main() {
    let mut cli = App::new("st_counts")
        .version("0.1.0")
        .author("Parker J. Rule <parker.rule@tufts.edu>")
        .about("Computes spanning tree counts from a pcompress file and a graph.")
        .arg(
            Arg::with_name("graph_path")
                .long("graph-json")
                .takes_value(true)
                .required(true)
                .help("The path of the dual graph (in NetworkX JSON format)."),
        )
        .arg(
            Arg::with_name("pop_col")
                .long("pop-col")
                .takes_value(true)
                .required(true)
                .help("The path of the population column in the dual graph."),
        );

    let matches = cli.get_matches();
    let graph_path = fs::canonicalize(PathBuf::from(matches.value_of("graph_path").unwrap()))
        .unwrap()
        .into_os_string()
        .into_string()
        .unwrap();
    let pop_col = matches.value_of("pop_col").unwrap();
    let graph = graph_from_networkx(&graph_path, &pop_col, Vec::<String>::new()).unwrap();

    let stdin = std::io::stdin();
    let mut reader = std::io::BufReader::with_capacity(usize::pow(2, 24), stdin.lock());
}
