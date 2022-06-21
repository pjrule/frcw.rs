//! Reversible ReCom distribution test for the frcw engine.
//!
//! Intended for use with the MGGG benchmark suite
//! (https://github.com/mggg/benchmarks).
use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use rand::SeedableRng;
use rand::rngs::SmallRng;
use clap::{value_t, App, Arg};
use frcw::graph::Graph;
use frcw::spanning_tree::{USTSampler, SpanningTreeSampler};
use frcw::buffers::SpanningTreeBuffer;

const SIDE: usize = 24;

fn main() {
    let matches = App::new("uniform-stats")
        .version("0.1.0")
        .author("Parker J. Rule <parker.rule@tufts.edu> and Jamie Tucker-Foltz")
        .about("Statistics of uniform spanning trees on a 24x24 grid graph")
        .arg(
            Arg::with_name("num_trees")
                .long("num-trees")
                .takes_value(true)
                .required(true)
                .help("The number of trees to draw."),
        )
        .arg(
            Arg::with_name("rng_seed")
                .long("rng-seed")
                .takes_value(true)
                .required(true)
                .help("The seed of the RNG used to draw proposals."),
        )
        .get_matches();

    let rng_seed = value_t!(matches.value_of("rng_seed"), u64).unwrap_or_else(|e| e.exit());
    let num_trees = value_t!(matches.value_of("num_trees"), usize).unwrap_or_else(|e| e.exit());
    let graph = Graph::rect_grid(SIDE, SIDE);
    let mut rng: SmallRng = SeedableRng::seed_from_u64(rng_seed);
    let mut sampler = USTSampler::new(SIDE * SIDE, &mut rng);
    let mut buffer = SpanningTreeBuffer::new(SIDE * SIDE);
    for _ in 1..num_trees {
        sampler.random_spanning_tree(&graph, &mut buffer, &mut rng);
    }
    // Done with benchmark.

    for _ in 1..num_trees {
        sampler.random_spanning_tree(&graph, &mut buffer, &mut rng);
        for (node_idx, neighbors) in buffer.st.iter().enumerate() {
            let root_c1 = node_idx / SIDE;
            let root_c2 = node_idx % SIDE;
            for neighbor in neighbors.iter() {
                let neighbor_c1 = neighbor / SIDE;
                let neighbor_c2 = neighbor % SIDE;
                println!("(({}, {}), ({}, {}))", root_c1, root_c2, neighbor_c1, neighbor_c2);
            }
        }
        println!("---");
    }

}
