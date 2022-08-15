use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use serde_json::json;
use serde::{Serialize, Deserialize};
use clap::{value_t, App, Arg};
use std::collections::HashMap;
use frcw::init::graph_from_networkx;
use petgraph::graph::{Graph, NodeIndex};
use serde_json::Value;
use std::collections::BTreeMap;
use std::io::Write;
use std::fs::{self, File};
use std::io::{self, BufRead};
use std::path::{Path, PathBuf};

// use jq_rs;


#[derive(Serialize,Deserialize,Clone,Debug)]
pub struct Districts {
    // pub plan_stats: HashMap<String, u64>,
    pub dist_stats: Vec<HashMap<String, u64>>
}

impl Districts {
    pub fn new() -> Districts {
        Districts {
            // plan_stats: HashMap::new(),
            dist_stats: vec![]
        }
    }
}

fn main() {
    let cli = App::new("jsonacc")
        .version("0.1.0")
        .author("Max Fan <root@max.fan>")
        .about("A simple jsonl accumulator intended to support jq queries");
    // let mut jq_query = jq_rs::compile(".dist_stats | map(.BVAP20 / .TOTPOP20)").unwrap();

    let mut districts = Districts::new();

    let stdin = std::io::stdin();
    let reader = std::io::BufReader::with_capacity(usize::pow(2, 24), stdin);

    let stdout = std::io::stdout();
    let mut writer = std::io::BufWriter::with_capacity(usize::pow(2, 24), stdout);

    for (line, input) in reader.lines().enumerate() {
        let contents = match input {
            Ok(content) => content,
            Err(_) => {
                panic!("Could not parse line {} as JSON", line);
            }
        };

        let line_data: Value = match serde_json::from_str(&contents) {
            Ok(data) => data,
            Err(_) => {
                panic!("Could not parse line {} as JSON", line);
            }
        };

        if let Some(init) = line_data.get("init") {
            if let Some(sums) = init.get("sums") {
                for (column, dist_values) in sums.as_object().unwrap() {
                    for (dist, dist_value) in dist_values.as_array().unwrap().iter().enumerate() {
                        if dist >= districts.dist_stats.len() {
                            districts.dist_stats.push(HashMap::new());
                        }
                        districts.dist_stats[dist].insert(column.to_string(), dist_value.as_u64().unwrap());
                    }
                }
            }
        }

        if let Some(step) = line_data.get("step") {
            if let Some(sums) = step.get("sums") {
                for (column, dist_values) in sums.as_object().unwrap() {
                    for (dist, dist_value) in dist_values.as_array().unwrap().iter().enumerate() {
                        districts.dist_stats[dist].insert(column.to_string(), dist_value.as_u64().unwrap());
                    }
                }
            }
        }

        if districts.dist_stats.len() != 0 {
            serde_json::to_writer(&mut writer, &districts);
            // writer.write(
            //     &jq_query.run(
            //         &serde_json::to_string(&districts)
            //         .unwrap()
            //     )
            //     .expect("Unable to run jq")
            //     .into_bytes()
            // );
        }
    }

    writer.flush();
}

