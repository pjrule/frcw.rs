use frcw::graph::Graph;
use frcw::init::from_networkx;
use frcw::partition::Partition;
/// Graph/initial partition fixtures for long-running tests.
use std::path::PathBuf;

/// The location of the graph JSON data w.r.t. the project manifest.
const GRAPH_FIXTURES_DIR: &str = "graphs";

/// Iowa counties adjacency graph.
/// See https://github.com/mggg-states/IA-shapefiles
const IA_FILENAME: &str = "IA_counties.json";
const IA_POP_COL: &str = "TOTPOP";
const IA_ASSIGNMENT_COL: &str = "CD";

/// Virginia precincts adjacency graph.
/// See https://github.com/mggg-states/VA-shapefiles
const VA_FILENAME: &str = "VA_precincts.json";
const VA_POP_COL: &str = "TOTPOP";
const VA_ASSIGNMENT_COL: &str = "CD_16";

/// Pennsylvania precincts adjacency graph from a draft
/// shapefile processed by Max Fan (@InnovativeInventor).
/// Seed plan generated with `recursive_tree_part` in GerryChain.
const PA_FILENAME: &str = "PA_draft_4.json";
const PA_POP_COL: &str = "TOTPOP";
const PA_ASSIGNMENT_COL: &str = "seed_1";

/// 6x6 grid graph (rook adjacency, equal node populations, stripes assignment).
const SIX_FILENAME: &str = "6x6.json";
const SIX_POP_COL: &str = "population";
const SIX_ASSIGNMENT_COL: &str = "district";

/// Loads a graph/partition fixture with no attributes.
pub fn default_fixture(key: &str) -> (Graph, Partition) {
    return fixture_with_attributes(key, Vec::new());
}

/// Loads a graph/partition fixture with extra attribute columns.
pub fn fixture_with_attributes(key: &str, columns: Vec<&str>) -> (Graph, Partition) {
    let (filename, pop_col, assignment_col) = match key {
        "IA" => (IA_FILENAME, IA_POP_COL, IA_ASSIGNMENT_COL),
        "VA" => (VA_FILENAME, VA_POP_COL, VA_ASSIGNMENT_COL),
        "PA" => (PA_FILENAME, PA_POP_COL, PA_ASSIGNMENT_COL),
        "6x6" => (SIX_FILENAME, SIX_POP_COL, SIX_ASSIGNMENT_COL),
        bad => panic!("Unknown graph fixture '{}'", bad),
    };

    // stable dir: see https://stackoverflow.com/a/30004252
    let mut full_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    full_path.push(GRAPH_FIXTURES_DIR);
    full_path.push(filename);
    // PathBuf -> String: see https://stackoverflow.com/a/42579588
    let path_str = full_path.into_os_string().into_string().unwrap();
    let columns_owned = columns.iter().map(|c| c.to_string()).collect();
    return from_networkx(&path_str, pop_col, assignment_col, columns_owned).unwrap();
}

// TODO: allow for alternate seeds and population columns.
