//! Main CLI for GerryQL.
use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use clap::{App, Arg};
// use frcw::init::graph_from_networkx;
use ndarray::prelude::*;
use ndarray::Array1;
use ndarray_stats::QuantileExt;
use petgraph::algo::toposort;
use petgraph::graph::{Graph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use serde_json::{json, Value};
use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;

fn main() {
    let cli = App::new("gerryql")
        .version("0.1.0")
        .author("Parker J. Rule <parker.rule@tufts.edu>")
        .about("A query engine for statistics on long Markov chain runs.")
        .arg(
            Arg::with_name("graph_json")
                .long("graph-json")
                .takes_value(true)
                .help("The path of the dual graph (in NetworkX format)."),
        )
        .arg(
            Arg::with_name("query_json")
                .long("query-json")
                .takes_value(true)
                .help("The path of the query definitions (triggers batch mode)."),
        )
        .arg(
            Arg::with_name("chain_data")
                .long("chain-data")
                .takes_value(true)
                .help("The path of the chain run data (in JSONL format)."),
        )
        .arg(
            Arg::with_name("n_threads")
                .long("n-threads")
                .takes_value(true)
                .default_value("1")
                .help("The number of threads to use."),
        );
    let matches = cli.get_matches();
    let query_path = match matches.value_of("query_json") {
        Some(v) => Some(PathBuf::from(v)),
        None => None,
    };

    match query_path {
        Some(path) => {
            // Batch mode.
            let query_file = File::open(path).unwrap();
            let query_reader = BufReader::new(query_file);
            let queries_outer: Value = serde_json::from_reader(query_reader).unwrap();
            let queries = queries_outer["queries"].as_object().unwrap();
            /*
            let chain_data_path = match matches.value_of("chain_data") {
                Some(v) => PathBuf::from(v),
                None => None,
            };
            */
            let mut query_keys = vec![];
            let mut graphs = vec![];
            let mut roots = vec![];
            for (k, e) in queries.iter() {
                let (graph, root) =
                    expr_to_graph(&parse_expr(e.as_str().unwrap().to_string()).unwrap());
                query_keys.push(k.to_string());
                graphs.push(graph);
                roots.push(root);
            }
            let (hists, first_occs, hist_slices) = collect_stats(&mut graphs, &roots).unwrap();
            let mut results = json!({});
            for (((key, key_hist), key_first_occs), key_hist_slices) in query_keys
                .iter()
                .zip(hists.iter())
                .zip(first_occs.iter())
                .zip(hist_slices.iter())
            {
                results[key] = json!({
                    "hist": key_hist,
                    "first_occs": key_first_occs,
                    "hist_slices": key_hist_slices
                });
            }
            println!("{}", results);
        }
        None => {
            // Interactive mode.
            let chain_data_path = PathBuf::from(matches.value_of("chain_data").unwrap());
            loop {
                print!("> ");
                let _ = io::stdout().flush();
                let mut buffer = String::new();
                let _ = io::stdin().read_line(&mut buffer);
                let expr = parse_expr(buffer);
                match expr {
                    Err(e) => println!("cannot parse expression: {:?}", e),
                    Ok(expr) => {
                        let (mut comp_graph, root) = expr_to_graph(&expr);
                        readlines(&chain_data_path, &mut comp_graph, root).unwrap();
                    }
                }
            }
        }
    }
}

/// Runs lines from a JSONL file through a compute graph.
fn readlines(
    data_path: &PathBuf,
    comp_graph: &mut QLComputeGraph,
    root: NodeIndex,
) -> Result<(), io::Error> {
    let order = update_order(&comp_graph).unwrap();
    let mut started = false;
    let mut col_vals = BTreeMap::<String, Vec<i64>>::new();

    let file = File::open(data_path)?;
    let reader = BufReader::new(file);
    for (line, contents) in reader.lines().enumerate() {
        let line_data: Value = match serde_json::from_str(&contents?) {
            Ok(data) => data,
            Err(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    format!("Could not parse line {} as JSON", line),
                ))
            }
        };
        if let Some(init_data) = line_data.get("init") {
            if started {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    format!("Double initialization on line {}", line),
                ));
            }
            let cols = init_data.as_object().unwrap()["sums"].as_object().unwrap();
            let num_dists = init_data.as_object().unwrap()["num_dists"]
                .as_u64()
                .unwrap();
            for (col, vals) in cols.iter() {
                col_vals.insert(
                    format!(".{}", col),
                    vals.as_array()
                        .unwrap()
                        .iter()
                        .map(|c| c.as_i64().unwrap())
                        .collect(),
                );
            }
            started = true;
            eval_graph(
                comp_graph,
                &order,
                &col_vals,
                &(0..num_dists).map(|v| v as usize).collect::<Vec<usize>>(),
            )
            .unwrap();
            println!("{}", fmt_node_value(&comp_graph[root].value));
        }

        if let Some(step_data) = line_data.get("step") {
            if !started {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    format!("Step before initialization on line {}", line),
                ));
            }
            let dists: Vec<usize> = step_data["dists"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| d.as_u64().unwrap() as usize)
                .collect();
            let step_sums = step_data["sums"].as_object().unwrap();
            for (col, delta_vals) in step_sums.iter() {
                let prev_vals = col_vals.get_mut(&format!(".{}", col)).unwrap();
                for (dist, val) in dists.iter().zip(delta_vals.as_array().unwrap().iter()) {
                    prev_vals[*dist] = val.as_i64().unwrap();
                }
            }
            eval_graph(comp_graph, &order, &col_vals, &dists).unwrap();
            println!("{}", fmt_node_value(&comp_graph[root].value));
        }
    }
    Ok(())
}

type Hist = HashMap<Int, Int>;
type HistCollection = Vec<Hist>;

/// Collects statistics over compute graphs.
fn collect_stats(
    comp_graphs: &mut [QLComputeGraph],
    roots: &[NodeIndex],
) -> Result<(HistCollection, HistCollection, Vec<HistCollection>), io::Error> {
    let orders: Vec<Vec<NodeIndex>> = comp_graphs
        .iter()
        .map(|graph| update_order(graph).unwrap())
        .collect();
    let mut started = false;
    let mut col_vals = BTreeMap::<String, Vec<i64>>::new();

    let mut cur_hist_slices = vec![Hist::new(); comp_graphs.len()];
    let mut hist_slices = vec![HistCollection::new(); comp_graphs.len()];

    let mut hists = vec![Hist::new(); comp_graphs.len()];
    let mut first_occs = vec![Hist::new(); comp_graphs.len()];

    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    let mut line = 0;
    while let Some(contents) = lines.next() {
        match contents {
            Err(_) => break, // EOF?
            Ok(ref v) => {
                if v == "" {
                    break; // EOF?
                }
            }
        };
        let line_data: Value = match serde_json::from_str(&contents?) {
            Ok(data) => data,
            Err(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    format!("Could not parse line {} as JSON", line),
                ))
            }
        };
        if let Some(init_data) = line_data.get("init") {
            if started {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    format!("Double initialization on line {}", line),
                ));
            }
            let cols = init_data.as_object().unwrap()["sums"].as_object().unwrap();
            let num_dists = init_data.as_object().unwrap()["num_dists"]
                .as_u64()
                .unwrap();
            for (col, vals) in cols.iter() {
                col_vals.insert(
                    format!(".{}", col),
                    vals.as_array()
                        .unwrap()
                        .iter()
                        .map(|c| c.as_i64().unwrap())
                        .collect(),
                );
            }
            started = true;
            for (((((graph, root), order), key_hist), key_first_occs), key_hist_slice) in
                comp_graphs
                    .iter_mut()
                    .zip(roots.iter())
                    .zip(orders.iter())
                    .zip(hists.iter_mut())
                    .zip(first_occs.iter_mut())
                    .zip(cur_hist_slices.iter_mut())
            {
                eval_graph(
                    graph,
                    order,
                    &col_vals,
                    &(0..num_dists).map(|v| v as usize).collect::<Vec<usize>>(),
                )
                .unwrap();

                // Update the histogram and first occurrences statistics.
                let root_val = match &graph[*root].value {
                    Some(QLValue::Int(v)) => *v,
                    Some(QLValue::Bool(v)) => *v as Int,
                    Some(QLValue::Float(_)) => panic!("Floating-point histograms not supported."),
                    Some(_) => panic!("Cannot collect histogram over non-scalar value."),
                    None => unreachable!(),
                };
                match key_hist.entry(root_val) {
                    Entry::Occupied(o) => *o.into_mut() += 1,
                    Entry::Vacant(v) => {
                        v.insert(1);
                    }
                };
                match key_hist_slice.entry(root_val) {
                    Entry::Occupied(o) => *o.into_mut() += 1,
                    Entry::Vacant(v) => {
                        v.insert(1);
                    }
                };
                key_first_occs.entry(root_val).or_insert(line as Int);
            }
        }

        if let Some(step_data) = line_data.get("step") {
            if !started {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    format!("Step before initialization on line {}", line),
                ));
            }
            let dists: Vec<usize> = step_data["dists"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| d.as_u64().unwrap() as usize)
                .collect();
            let step_sums = step_data["sums"].as_object().unwrap();
            for (col, delta_vals) in step_sums.iter() {
                let prev_vals = col_vals.get_mut(&format!(".{}", col)).unwrap();
                for (dist, val) in dists.iter().zip(delta_vals.as_array().unwrap().iter()) {
                    prev_vals[*dist] = val.as_i64().unwrap();
                }
            }
            for (((((graph, root), order), key_hist), key_first_occs), key_hist_slice) in
                comp_graphs
                    .iter_mut()
                    .zip(roots.iter())
                    .zip(orders.iter())
                    .zip(hists.iter_mut())
                    .zip(first_occs.iter_mut())
                    .zip(cur_hist_slices.iter_mut())
            {
                eval_graph(graph, order, &col_vals, &dists).unwrap();

                // Update the histogram and first occurrences statistics.
                let root_val = match &graph[*root].value {
                    Some(QLValue::Int(v)) => *v,
                    Some(QLValue::Bool(v)) => *v as Int,
                    Some(QLValue::Float(_)) => panic!("Floating-point histograms not supported."),
                    Some(_) => panic!("Cannot collect histogram over non-scalar value."),
                    None => unreachable!(),
                };
                match key_hist.entry(root_val) {
                    Entry::Occupied(o) => *o.into_mut() += 1,
                    Entry::Vacant(v) => {
                        v.insert(1);
                    }
                };
                match key_hist_slice.entry(root_val) {
                    Entry::Occupied(o) => *o.into_mut() += 1,
                    Entry::Vacant(v) => {
                        v.insert(1);
                    }
                };
                key_first_occs.entry(root_val).or_insert(line as Int);
            }
        }
        if line > 0 && line % 20000 == 0 {
            for (key_hist_slice, key_hist_slices) in
                cur_hist_slices.iter_mut().zip(hist_slices.iter_mut())
            {
                key_hist_slices.push(key_hist_slice.clone());
                key_hist_slice.clear();
            }
        }
        line += 1;
    }
    Ok((hists, first_occs, hist_slices))
}

/// GerryQL basis functions.
#[derive(Copy, Clone, PartialEq, Debug)]
enum QLPrimitive {
    Add,
    Sub,
    Mult,
    Divide,
    // TODO: IntDivide, Mod, Dot
    Sort,
    Sum,
    Min,
    Max,
    Median,
    Mode,
    Mean,
    Eq,
    Gt,
    Geq,
    Lt,
    Leq,
    And,
    Or,
    Not,
    Zeros,
}

impl fmt::Display for QLPrimitive {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let token = match self {
            QLPrimitive::Add => "+",
            QLPrimitive::Sub => "-",
            QLPrimitive::Mult => "*",
            QLPrimitive::Divide => "/",
            QLPrimitive::Sort => "sort",
            QLPrimitive::Sum => "sum",
            QLPrimitive::Min => "min",
            QLPrimitive::Max => "max",
            QLPrimitive::Median => "median",
            QLPrimitive::Mode => "mode",
            QLPrimitive::Mean => "mean",
            QLPrimitive::Eq => "=",
            QLPrimitive::Gt => ">",
            QLPrimitive::Geq => ">=",
            QLPrimitive::Lt => "<",
            QLPrimitive::Leq => "<=",
            QLPrimitive::And => "and",
            QLPrimitive::Or => "or",
            QLPrimitive::Not => "not",
            QLPrimitive::Zeros => "zeros",
        };
        write!(f, "{}", token)
    }
}

/// GerryQL expressions.
#[derive(Clone, PartialEq, Debug)]
enum QLExpr {
    Bool(bool),
    Int(Int),
    Float(Float),
    Var(String),
    Column(String),
    Primitive(QLPrimitive),
    Lambda(Vec<String>, Box<QLExpr>),
    Apply(Box<QLExpr>, Vec<QLExpr>),
}

impl fmt::Display for QLExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let expr = match self {
            QLExpr::Bool(b) => format!("{}", b),
            QLExpr::Int(i) => format!("{}", i),
            QLExpr::Float(f) => format!("{}", f),
            QLExpr::Column(c) => c.to_string(),
            QLExpr::Primitive(prim) => format!("{}", prim),
            QLExpr::Var(v) => v.to_string(),
            // TODO: format non-implicit args.
            QLExpr::Lambda(_, body) => format!("#({})", body),
            QLExpr::Apply(expr, args) => {
                let arg_fmts: Vec<String> = args.iter().map(|arg| format!("{}", arg)).collect();
                let args_formatted = arg_fmts.join(" ");
                format!("({} {})", expr, args_formatted)
            }
        };
        write!(f, "{}", expr)
    }
}

/// Integer type used in interpreter.
type Int = i64;
/// Float type used in interpreter.
type Float = f64;
/// Integer array type used in interpreter.
type IntArray = Array1<Int>;
/// Float array type used in interpreter.
type FloatArray = Array1<Float>;
/// Bool array type used in interpreter.
type BoolArray = Array1<bool>;

/// GerryQL values.
#[derive(Clone, PartialEq, Debug)]
enum QLValue {
    Bool(bool),
    Int(Int),
    Float(Float),
    BoolArray(BoolArray),
    IntArray(IntArray),
    FloatArray(FloatArray),
}

/// GerryQL errors.
#[derive(Clone, PartialEq, Debug)]
enum QLError {
    UnexpectedEOF,
    UnmatchedParens,
    ExtraChars,
    UnreachableState,
    AppliedNonFunction,
    BadToken(String),
}

//// Parsed GerryQL tokens.
#[derive(PartialEq)]
enum QLToken {
    StartLambda,
    Start,
    End,
    BoolLiteral(bool),
    IntLiteral(Int),
    FloatLiteral(Float),
    ColumnLiteral(String),
    PrimitiveName(QLPrimitive),
    Name(String),
    Unknown,
}

fn tokenize(raw_expr: String) -> Vec<String> {
    raw_expr
        .replace("(", " ( ")
        .replace(")", " ) ")
        .replace("[", " [ ")
        .replace("]", " ] ")
        .split_whitespace()
        .map(|t| t.to_string())
        .collect()
}

/// Converts a token to a primitive function identifier, if possible.
fn token_to_primitive(token: &str) -> Option<QLPrimitive> {
    match token {
        "+" => Some(QLPrimitive::Add),
        "-" => Some(QLPrimitive::Sub),
        "*" => Some(QLPrimitive::Mult),
        "/" => Some(QLPrimitive::Divide),
        "sort" => Some(QLPrimitive::Sort),
        "sum" => Some(QLPrimitive::Sum),
        "min" => Some(QLPrimitive::Min),
        "max" => Some(QLPrimitive::Max),
        "median" => Some(QLPrimitive::Median),
        "mode" => Some(QLPrimitive::Mode),
        "mean" => Some(QLPrimitive::Mean),
        "=" => Some(QLPrimitive::Eq),
        ">" => Some(QLPrimitive::Gt),
        ">=" => Some(QLPrimitive::Geq),
        "<" => Some(QLPrimitive::Lt),
        "<=" => Some(QLPrimitive::Leq),
        "and" => Some(QLPrimitive::And),
        "or" => Some(QLPrimitive::Or),
        "not" => Some(QLPrimitive::Not),
        "&&" => Some(QLPrimitive::And),
        "||" => Some(QLPrimitive::Or),
        "zeros" => Some(QLPrimitive::Zeros),
        _ => None,
    }
}

fn parse_token(token: &str) -> QLToken {
    if let Some(prim) = token_to_primitive(token) {
        return QLToken::PrimitiveName(prim);
    }
    if let Ok(val) = token.parse::<Int>() {
        return QLToken::IntLiteral(val);
    }
    if let Ok(val) = token.parse::<Float>() {
        return QLToken::FloatLiteral(val);
    }
    if token.len() > 1 {
        if let Some('.') = token.chars().next() {
            return QLToken::ColumnLiteral(token.to_owned());
        }
    }
    match token {
        "(" => QLToken::Start,
        "[" => QLToken::Start,
        ")" => QLToken::End,
        "]" => QLToken::End,
        "#" => QLToken::StartLambda,
        "true" => QLToken::BoolLiteral(true),
        "false" => QLToken::BoolLiteral(false),
        name => QLToken::Name(name.to_string()), // TODO: what names are allowable?
    }
}

fn parse_next_expr(tokens: &[String]) -> Result<(QLExpr, &[String]), QLError> {
    if tokens.len() == 0 {
        return Err(QLError::UnexpectedEOF);
    }
    let token = &tokens[0];
    let tt = parse_token(&token);
    if tt == QLToken::StartLambda {
        // We expect at least three tokens in the stream:
        // [open lambda] [open] [something] [close]
        if tokens.len() < 4 {
            return Err(QLError::UnexpectedEOF);
        }
        if parse_token(&tokens[1]) != QLToken::Start {
            return Err(QLError::BadToken(tokens[1].to_owned()));
        }
        let (body, rest) = parse_next_expr(&tokens[2..])?;
        if rest.is_empty() {
            return Err(QLError::UnexpectedEOF);
        }
        let end_token = parse_token(&rest[0]);
        if end_token != QLToken::End {
            return Err(QLError::BadToken(rest[0].to_owned()));
        }
        // TODO: allow arguments!
        return Ok((QLExpr::Lambda(vec![], Box::new(body)), &rest[1..]));
    } else if tt == QLToken::Start {
        // We expect at least three tokens in the stream:
        // [open] [something] [close]
        if tokens.len() < 3 {
            return Err(QLError::UnexpectedEOF);
        }
        let (applied, mut rest) = parse_next_expr(&tokens[1..])?;
        match applied {
            QLExpr::Primitive(_) => (),
            QLExpr::Var(_) => (),
            QLExpr::Lambda(_, _) => (),
            _ => return Err(QLError::AppliedNonFunction),
        };

        let mut args: Vec<QLExpr> = vec![];
        while !rest.is_empty() {
            let next_token = parse_token(&rest[0]);
            if next_token == QLToken::End {
                // Done reading arguments.
                break;
            }
            let (arg, next_rest) = parse_next_expr(rest)?;
            rest = next_rest;
            args.push(arg);
        }

        if rest.is_empty() {
            return Err(QLError::UnexpectedEOF);
        }
        let end_token = parse_token(&rest[0]);
        if end_token != QLToken::End {
            return Err(QLError::BadToken(rest[0].to_owned()));
        }
        return Ok((QLExpr::Apply(Box::new(applied), args), &rest[1..]));
    }

    let rest = &tokens[1..];
    match tt {
        QLToken::BoolLiteral(val) => Ok((QLExpr::Bool(val), rest)),
        QLToken::FloatLiteral(val) => Ok((QLExpr::Float(val), rest)),
        QLToken::IntLiteral(val) => Ok((QLExpr::Int(val), rest)),
        QLToken::ColumnLiteral(val) => Ok((QLExpr::Column(val), rest)),
        QLToken::PrimitiveName(prim) => Ok((QLExpr::Primitive(prim), rest)),
        QLToken::Name(name) => Ok((QLExpr::Var(name), rest)),
        QLToken::End => Err(QLError::UnexpectedEOF),
        QLToken::Unknown => Err(QLError::BadToken(token.to_owned())),
        _ => Err(QLError::UnreachableState),
    }
}

/// Attempts to parse a GerryQL expression from a raw string representation.
fn parse_expr(raw_expr: String) -> Result<QLExpr, QLError> {
    let tokens = tokenize(raw_expr);
    match parse_next_expr(&tokens) {
        Err(e) => Err(e),
        Ok((exp, rest)) => {
            if rest.len() > 0 {
                if rest.iter().all(|t| parse_token(t) == QLToken::End) {
                    Err(QLError::UnmatchedParens)
                } else {
                    Err(QLError::ExtraChars)
                }
            } else {
                Ok(exp)
            }
        }
    }
}

/// What kind of computation does the data flow graph represent?
#[derive(Clone, PartialEq)]
enum QLNodeKind {
    Primitive(QLPrimitive),
    Constant,
    NotImplementedYet,
    Column(String),
}

impl fmt::Display for QLNodeKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let kind = match self {
            QLNodeKind::Primitive(prim) => format!("primitive fn ({})", prim),
            QLNodeKind::Constant => "constant".to_string(),
            QLNodeKind::NotImplementedYet => "???".to_string(),
            QLNodeKind::Column(col) => format!("column data ({})", col),
        };
        write!(f, "{}", kind)
    }
}

/// How should a computation's value be updated?
enum QLUpdate {
    /// Any time a tracked property of a districting plan changes,
    /// update the whole node.
    All,
    /// The node contains a vector of values with length equal
    /// to the number of districts expected in a plan. We can
    /// safely update district-level statistics without updating
    /// every entry in the vector.
    PerDistrict,
}

/// A node in a GerryQL expression's computation/dependency graph.
struct QLComputeNode {
    /// The kind of the compute node.
    kind: QLNodeKind,
    /// How the compute node should be updated.
    update: QLUpdate,
    /// The compute node's cached value.
    value: Option<QLValue>,
    /// The hash of the expression that the compute node is the
    /// root of. The container is responsible for handling collisions.
    expr_hash: u64,
}

fn fmt_node_value(val: &Option<QLValue>) -> String {
    match val {
        None => "()".to_string(),
        Some(QLValue::Int(v)) => format!("{} : int", v),
        Some(QLValue::Bool(v)) => format!("{} : bool", v),
        Some(QLValue::Float(v)) => format!("{} : float", v),
        Some(QLValue::IntArray(v)) => format!("{} : int[{}]", v, v.len()),
        Some(QLValue::BoolArray(v)) => format!("{} : bool[{}]", v, v.len()),
        Some(QLValue::FloatArray(v)) => format!("{} : float[{}]", v, v.len()),
    }
}

impl fmt::Display for QLComputeNode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} (val = {:?})", self.kind, fmt_node_value(&self.value))
    }
}

/// Edges in the dependency graphs are labeled with argument IDs.
type QLComputeEdge = usize;

type QLComputeGraph = Graph<QLComputeNode, QLComputeEdge>;

/// Generates a computation graph from a GerryQL expression.
///
/// Returns the compute graph and the index of the node corresponding
/// to the expression.
fn expr_to_graph(expr: &QLExpr) -> (QLComputeGraph, NodeIndex) {
    let mut graph = QLComputeGraph::new();
    let node_idx = expr_to_node(expr, &mut graph, None, None);
    (graph, node_idx)
}

/// Recursively builds a computation graph from a GerryQL expression.
fn expr_to_node(
    expr: &QLExpr,
    graph: &mut QLComputeGraph,
    parent: Option<NodeIndex>,
    child_id: Option<QLComputeEdge>,
) -> NodeIndex {
    // Convert the expression to compute graph metadata.
    let (kind, value, deps) = match expr {
        QLExpr::Bool(v) => (
            QLNodeKind::Constant,
            Some(QLValue::Bool(v.to_owned())),
            None,
        ),
        QLExpr::Int(v) => (QLNodeKind::Constant, Some(QLValue::Int(v.to_owned())), None),
        QLExpr::Float(v) => (
            QLNodeKind::Constant,
            Some(QLValue::Float(v.to_owned())),
            None,
        ),
        QLExpr::Apply(exp, args) => match **exp {
            QLExpr::Primitive(prim) => (QLNodeKind::Primitive(prim.to_owned()), None, Some(args)),
            _ => (QLNodeKind::NotImplementedYet, None, Some(args)),
        },
        QLExpr::Column(col) => (QLNodeKind::Column(col.to_owned()), None, None),
        // TODO
        _ => (QLNodeKind::NotImplementedYet, None, None),
    };
    let node_idx = graph.add_node(QLComputeNode {
        kind: kind,
        value: value,
        update: QLUpdate::All, // Be conservative initially.
        expr_hash: 0,          // TODO
    });

    // Add a dependency edge from the new node to the parent, if there's
    // a parent available.
    if let Some(parent_idx) = parent {
        graph.add_edge(node_idx, parent_idx, child_id.unwrap());
    }

    // Create nodes and edges for the subexpressions (dependencies).
    if let Some(children) = deps {
        for (dep_id, dep) in children.iter().enumerate() {
            expr_to_node(dep, graph, Some(node_idx), Some(dep_id));
        }
    }
    node_idx
}

/// Coerces a boolean to a float, matching Python semantics.
fn bool_to_float(v: bool) -> Float {
    match v {
        false => 0.0,
        true => 1.0,
    }
}

macro_rules! shape_check {
    ($lhs:expr, $rhs:expr, $op:expr, $out:expr) => {{
        let lhs_shape = $lhs.shape();
        let rhs_shape = $rhs.shape();
        if lhs_shape == rhs_shape {
            Ok($out($op($lhs, $rhs)))
        } else {
            // TODO: multidimensional arrays?
            Err(format!(
                "operands could not be broadcast together with lengths {}, {}",
                lhs_shape[0], rhs_shape[0]
            ))
        }
    }};
}

macro_rules! bin_broadcast_op {
    ($lhs:expr, $rhs:expr, $op:expr) => {
        match ($lhs, $rhs) {
            // int * X
            (Some(QLValue::Int(l)), Some(QLValue::Int(r))) => Ok(QLValue::Int($op(l, r))),
            (Some(QLValue::Int(l)), Some(QLValue::Bool(r))) => Ok(QLValue::Int($op(l, *r as Int))),
            (Some(QLValue::Int(l)), Some(QLValue::Float(r))) => {
                Ok(QLValue::Float($op(*l as Float, r)))
            }
            (Some(QLValue::Int(l)), Some(QLValue::IntArray(r))) => {
                Ok(QLValue::IntArray($op(*l, r)))
            }
            (Some(QLValue::Int(l)), Some(QLValue::BoolArray(r))) => {
                Ok(QLValue::IntArray(r.map(|v| $op(l, *v as Int))))
            }
            (Some(QLValue::Int(l)), Some(QLValue::FloatArray(r))) => {
                Ok(QLValue::FloatArray(r.map(|v| $op(*l as Float, *v))))
            }
            // bool * X
            (Some(QLValue::Bool(l)), Some(QLValue::Int(r))) => Ok(QLValue::Int($op(*l as Int, r))),
            (Some(QLValue::Bool(l)), Some(QLValue::Bool(r))) => {
                Ok(QLValue::Int($op(*l as Int, *r as Int)))
            }
            (Some(QLValue::Bool(l)), Some(QLValue::Float(r))) => {
                Ok(QLValue::Float($op(bool_to_float(*l), r)))
            }
            (Some(QLValue::Bool(l)), Some(QLValue::IntArray(r))) => {
                Ok(QLValue::IntArray($op(*l as Int, r)))
            }
            (Some(QLValue::Bool(l)), Some(QLValue::BoolArray(r))) => {
                Ok(QLValue::IntArray(r.map(|v| $op(*l as Int, *v as Int))))
            }
            (Some(QLValue::Bool(l)), Some(QLValue::FloatArray(r))) => {
                Ok(QLValue::FloatArray($op(bool_to_float(*l), r)))
            }
            // float * X
            (Some(QLValue::Float(l)), Some(QLValue::Int(r))) => {
                Ok(QLValue::Float($op(l, *r as Float)))
            }
            (Some(QLValue::Float(l)), Some(QLValue::Bool(r))) => {
                Ok(QLValue::Float($op(l, bool_to_float(*r))))
            }
            (Some(QLValue::Float(l)), Some(QLValue::Float(r))) => Ok(QLValue::Float($op(l, r))),
            (Some(QLValue::Float(l)), Some(QLValue::IntArray(r))) => {
                Ok(QLValue::FloatArray(r.map(|v| $op(l, *v as Float))))
            }
            (Some(QLValue::Float(l)), Some(QLValue::BoolArray(r))) => {
                Ok(QLValue::FloatArray(r.map(|v| $op(l, bool_to_float(*v)))))
            }
            (Some(QLValue::Float(l)), Some(QLValue::FloatArray(r))) => {
                Ok(QLValue::FloatArray($op(*l, r)))
            }
            // int array * X
            (Some(QLValue::IntArray(l)), Some(QLValue::Int(r))) => {
                Ok(QLValue::IntArray($op(l, *r)))
            }
            (Some(QLValue::IntArray(l)), Some(QLValue::Bool(r))) => {
                Ok(QLValue::IntArray($op(l, *r as Int)))
            }
            (Some(QLValue::IntArray(l)), Some(QLValue::Float(r))) => {
                Ok(QLValue::FloatArray(l.map(|v| $op(*v as Float, r))))
            }
            (Some(QLValue::IntArray(l)), Some(QLValue::IntArray(r))) => {
                shape_check!(
                    l,
                    r,
                    |l: &IntArray, r: &IntArray| $op(l, r),
                    QLValue::IntArray
                )
            }
            (Some(QLValue::IntArray(l)), Some(QLValue::BoolArray(r))) => {
                shape_check!(
                    l,
                    r,
                    |l: &IntArray, r: &BoolArray| $op(l, r.map(|v| *v as Int)),
                    QLValue::IntArray
                )
            }
            (Some(QLValue::IntArray(l)), Some(QLValue::FloatArray(r))) => {
                shape_check!(
                    l,
                    r,
                    |l: &IntArray, r: &FloatArray| $op(l.map(|v| *v as Float), r),
                    QLValue::FloatArray
                )
            }
            // bool array * X
            (Some(QLValue::BoolArray(l)), Some(QLValue::Int(r))) => {
                Ok(QLValue::IntArray(l.map(|v| $op(*v as Int, r))))
            }
            (Some(QLValue::BoolArray(l)), Some(QLValue::Bool(r))) => {
                Ok(QLValue::IntArray(l.map(|v| $op(*v as Int, *r as Int))))
            }
            (Some(QLValue::BoolArray(l)), Some(QLValue::Float(r))) => {
                Ok(QLValue::FloatArray(l.map(|v| $op(bool_to_float(*v), r))))
            }
            (Some(QLValue::BoolArray(l)), Some(QLValue::IntArray(r))) => {
                shape_check!(
                    l,
                    r,
                    |l: &BoolArray, r: &IntArray| $op(l.map(|v| *v as Int), r),
                    QLValue::IntArray
                )
            }
            (Some(QLValue::BoolArray(l)), Some(QLValue::BoolArray(r))) => {
                shape_check!(
                    l,
                    r,
                    |l: &BoolArray, r: &BoolArray| $op(l.map(|v| *v as Int), r.map(|v| *v as Int)),
                    QLValue::IntArray
                )
            }
            (Some(QLValue::BoolArray(l)), Some(QLValue::FloatArray(r))) => {
                shape_check!(
                    l,
                    r,
                    |l: &BoolArray, r: &FloatArray| $op(l.map(|v| bool_to_float(*v)), r),
                    QLValue::FloatArray
                )
            }
            // float array * X
            (Some(QLValue::FloatArray(l)), Some(QLValue::Int(r))) => {
                Ok(QLValue::FloatArray($op(l, *r as Float)))
            }
            (Some(QLValue::FloatArray(l)), Some(QLValue::Bool(r))) => {
                Ok(QLValue::FloatArray($op(l, bool_to_float(*r))))
            }
            (Some(QLValue::FloatArray(l)), Some(QLValue::Float(r))) => {
                Ok(QLValue::FloatArray($op(l, *r)))
            }
            (Some(QLValue::FloatArray(l)), Some(QLValue::IntArray(r))) => {
                shape_check!(
                    l,
                    r,
                    |l: &FloatArray, r: &IntArray| $op(l, r.map(|v| *v as Float)),
                    QLValue::FloatArray
                )
            }
            (Some(QLValue::FloatArray(l)), Some(QLValue::BoolArray(r))) => {
                shape_check!(
                    l,
                    r,
                    |l: &FloatArray, r: &BoolArray| $op(l, r.map(|v| bool_to_float(*v)),),
                    QLValue::FloatArray
                )
            }
            (Some(QLValue::FloatArray(l)), Some(QLValue::FloatArray(r))) => {
                shape_check!(
                    l,
                    r,
                    |l: &FloatArray, r: &FloatArray| $op(l, r),
                    QLValue::FloatArray
                )
            }
            _ => unreachable!(),
        }
    };
}

macro_rules! cmp_broadcast_op {
    ($lhs:expr, $rhs:expr, $op:expr) => {
        match ($lhs, $rhs) {
            // int * X
            (Some(QLValue::Int(l)), Some(QLValue::Int(r))) => Ok(QLValue::Bool($op(*l, *r))),
            (Some(QLValue::Int(l)), Some(QLValue::Bool(r))) => {
                Ok(QLValue::Bool($op(*l, *r as Int)))
            }
            (Some(QLValue::Int(l)), Some(QLValue::Float(r))) => {
                Ok(QLValue::Bool($op(*l as Float, *r)))
            }
            (Some(QLValue::Int(l)), Some(QLValue::IntArray(r))) => {
                Ok(QLValue::BoolArray(r.map(|v| $op(*l, *v))))
            }
            (Some(QLValue::Int(l)), Some(QLValue::BoolArray(r))) => {
                Ok(QLValue::BoolArray(r.map(|v| $op(*l, *v as Int))))
            }
            (Some(QLValue::Int(l)), Some(QLValue::FloatArray(r))) => {
                Ok(QLValue::BoolArray(r.map(|v| $op(*l as Float, *v))))
            }
            // bool * X
            (Some(QLValue::Bool(l)), Some(QLValue::Int(r))) => {
                Ok(QLValue::Bool($op(*l as Int, *r)))
            }
            (Some(QLValue::Bool(l)), Some(QLValue::Bool(r))) => {
                Ok(QLValue::Bool($op(*l as Int, *r as Int)))
            }
            (Some(QLValue::Bool(l)), Some(QLValue::Float(r))) => {
                Ok(QLValue::Bool($op(bool_to_float(*l), *r)))
            }
            (Some(QLValue::Bool(l)), Some(QLValue::IntArray(r))) => {
                Ok(QLValue::BoolArray(r.map(|v| $op(*l as Int, *v))))
            }
            (Some(QLValue::Bool(l)), Some(QLValue::BoolArray(r))) => {
                Ok(QLValue::BoolArray(r.map(|v| $op(*l as Int, *v as Int))))
            }
            (Some(QLValue::Bool(l)), Some(QLValue::FloatArray(r))) => {
                Ok(QLValue::BoolArray(r.map(|v| $op(bool_to_float(*l), *v))))
            }
            // float * X
            (Some(QLValue::Float(l)), Some(QLValue::Int(r))) => {
                Ok(QLValue::Bool($op(*l, *r as Float)))
            }
            (Some(QLValue::Float(l)), Some(QLValue::Bool(r))) => {
                Ok(QLValue::Bool($op(*l, bool_to_float(*r))))
            }
            (Some(QLValue::Float(l)), Some(QLValue::Float(r))) => Ok(QLValue::Bool($op(*l, *r))),
            (Some(QLValue::Float(l)), Some(QLValue::IntArray(r))) => {
                Ok(QLValue::BoolArray(r.map(|v| $op(*l, *v as Float))))
            }
            (Some(QLValue::Float(l)), Some(QLValue::BoolArray(r))) => {
                Ok(QLValue::BoolArray(r.map(|v| $op(*l, bool_to_float(*v)))))
            }
            (Some(QLValue::Float(l)), Some(QLValue::FloatArray(r))) => {
                Ok(QLValue::BoolArray(r.map(|v| $op(*l, *v))))
            }
            // int array * X
            (Some(QLValue::IntArray(l)), Some(QLValue::Int(r))) => {
                Ok(QLValue::BoolArray(l.map(|v| $op(*v, *r))))
            }
            (Some(QLValue::IntArray(l)), Some(QLValue::Bool(r))) => {
                Ok(QLValue::BoolArray(l.map(|v| $op(*v, *r as Int))))
            }
            (Some(QLValue::IntArray(l)), Some(QLValue::Float(r))) => {
                Ok(QLValue::BoolArray(l.map(|v| $op(*v as Float, *r))))
            }
            (Some(QLValue::IntArray(l)), Some(QLValue::IntArray(r))) => {
                shape_check!(
                    l,
                    r,
                    |l: &IntArray, r: &IntArray| l
                        .iter()
                        .zip(r.iter())
                        .map(|(v1, v2)| $op(*v1, *v2))
                        .collect(),
                    QLValue::BoolArray
                )
            }
            (Some(QLValue::IntArray(l)), Some(QLValue::BoolArray(r))) => {
                shape_check!(
                    l,
                    r,
                    |l: &IntArray, r: &BoolArray| l
                        .iter()
                        .zip(r.iter())
                        .map(|(v1, v2)| $op(*v1, *v2 as Int))
                        .collect(),
                    QLValue::BoolArray
                )
            }
            (Some(QLValue::IntArray(l)), Some(QLValue::FloatArray(r))) => {
                shape_check!(
                    l,
                    r,
                    |l: &IntArray, r: &FloatArray| l
                        .iter()
                        .zip(r.iter())
                        .map(|(v1, v2)| $op(*v1 as Float, *v2))
                        .collect(),
                    QLValue::BoolArray
                )
            }
            // bool array * X
            (Some(QLValue::BoolArray(l)), Some(QLValue::Int(r))) => {
                Ok(QLValue::BoolArray(l.map(|v| $op(*v as Int, *r))))
            }
            (Some(QLValue::BoolArray(l)), Some(QLValue::Bool(r))) => {
                Ok(QLValue::BoolArray(l.map(|v| $op(*v as Int, *r as Int))))
            }
            (Some(QLValue::BoolArray(l)), Some(QLValue::Float(r))) => {
                Ok(QLValue::BoolArray(l.map(|v| $op(bool_to_float(*v), *r))))
            }
            (Some(QLValue::BoolArray(l)), Some(QLValue::IntArray(r))) => {
                shape_check!(
                    l,
                    r,
                    |l: &BoolArray, r: &IntArray| l
                        .iter()
                        .zip(r.iter())
                        .map(|(v1, v2)| $op(*v1 as Int, *v2))
                        .collect(),
                    QLValue::BoolArray
                )
            }
            (Some(QLValue::BoolArray(l)), Some(QLValue::BoolArray(r))) => {
                shape_check!(
                    l,
                    r,
                    |l: &BoolArray, r: &BoolArray| l
                        .iter()
                        .zip(r.iter())
                        .map(|(v1, v2)| $op(*v1 as Int, *v2 as Int))
                        .collect(),
                    QLValue::BoolArray
                )
            }
            (Some(QLValue::BoolArray(l)), Some(QLValue::FloatArray(r))) => {
                shape_check!(
                    l,
                    r,
                    |l: &BoolArray, r: &FloatArray| l
                        .iter()
                        .zip(r.iter())
                        .map(|(v1, v2)| $op(bool_to_float(*v1), *v2))
                        .collect(),
                    QLValue::BoolArray
                )
            }
            // float array * X
            (Some(QLValue::FloatArray(l)), Some(QLValue::Int(r))) => {
                Ok(QLValue::BoolArray(l.map(|v| $op(*v, *r as Float))))
            }
            (Some(QLValue::FloatArray(l)), Some(QLValue::Bool(r))) => {
                Ok(QLValue::BoolArray(l.map(|v| $op(*v, bool_to_float(*r)))))
            }
            (Some(QLValue::FloatArray(l)), Some(QLValue::Float(r))) => {
                Ok(QLValue::BoolArray(l.map(|v| $op(*v, *r))))
            }
            (Some(QLValue::FloatArray(l)), Some(QLValue::IntArray(r))) => {
                shape_check!(
                    l,
                    r,
                    |l: &FloatArray, r: &IntArray| l
                        .iter()
                        .zip(r.iter())
                        .map(|(v1, v2)| $op(*v1, *v2 as Float))
                        .collect(),
                    QLValue::BoolArray
                )
            }
            (Some(QLValue::FloatArray(l)), Some(QLValue::BoolArray(r))) => {
                shape_check!(
                    l,
                    r,
                    |l: &FloatArray, r: &BoolArray| l
                        .iter()
                        .zip(r.iter())
                        .map(|(v1, v2)| $op(*v1, bool_to_float(*v2)))
                        .collect(),
                    QLValue::BoolArray
                )
            }
            (Some(QLValue::FloatArray(l)), Some(QLValue::FloatArray(r))) => {
                shape_check!(
                    l,
                    r,
                    |l: &FloatArray, r: &FloatArray| l
                        .iter()
                        .zip(r.iter())
                        .map(|(v1, v2)| $op(*v1, *v2))
                        .collect(),
                    QLValue::BoolArray
                )
            }
            _ => unreachable!(),
        }
    };
}

/// Casts a generic QLValue to a boolean (array).
fn bool_cast(v: Option<&QLValue>) -> Option<QLValue> {
    match v {
        Some(QLValue::Int(v)) => Some(QLValue::Bool(*v != 0)),
        Some(QLValue::Bool(v)) => Some(QLValue::Bool(*v)),
        Some(QLValue::Float(v)) => Some(QLValue::Bool(*v != 0.0)),
        Some(QLValue::IntArray(v)) => Some(QLValue::BoolArray(v.map(|e| *e != 0))),
        Some(QLValue::BoolArray(v)) => Some(QLValue::BoolArray(v.clone())), // TODO: inefficient?
        Some(QLValue::FloatArray(v)) => Some(QLValue::BoolArray(v.map(|e| *e != 0.0))),
        None => None,
    }
}

macro_rules! bool_broadcast_op {
    ($lhs:expr, $rhs:expr, $op:expr) => {
        match (bool_cast($lhs), bool_cast($rhs)) {
            (Some(QLValue::Bool(l)), Some(QLValue::Bool(r))) => Ok(QLValue::Bool($op(l, r))),
            (Some(QLValue::Bool(l)), Some(QLValue::BoolArray(r))) => {
                Ok(QLValue::BoolArray(r.map(|v| $op(l, *v))))
            }
            (Some(QLValue::BoolArray(l)), Some(QLValue::Bool(r))) => {
                Ok(QLValue::BoolArray(l.map(|v| $op(*v, r))))
            }
            (Some(QLValue::BoolArray(l)), Some(QLValue::BoolArray(r))) => {
                shape_check!(
                    &l,
                    &r,
                    |l: &BoolArray, r: &BoolArray| l
                        .iter()
                        .zip(r.iter())
                        .map(|(v1, v2)| $op(*v1, *v2))
                        .collect(),
                    QLValue::BoolArray
                )
            }
            _ => unreachable!(),
        }
    };
}

/// Fills in all values in a computation graph.
fn eval_graph(
    graph: &mut QLComputeGraph,
    order: &[NodeIndex],
    column_sums: &BTreeMap<String, Vec<i64>>,
    dist_diff: &[usize],
) -> Result<(), String> {
    for &node_id in order.iter() {
        match &graph[node_id].kind {
            QLNodeKind::Column(col) => {
                // Update the column sums cached in the compute graph.
                let column_vals = column_sums.get(col);
                match column_vals {
                    Some(vals) => {
                        match &mut graph[node_id].value {
                            None => {
                                // Initialize an array in the graph for the column.
                                graph[node_id].value = Some(QLValue::IntArray(Array1::from_vec(
                                    vals.iter().map(|&v| v as Int).collect(),
                                )));
                            }
                            Some(QLValue::IntArray(arr)) => {
                                // Update the entries in an existing array.
                                for &dist_idx in dist_diff.iter() {
                                    arr[dist_idx] = vals[dist_idx] as Int;
                                }
                            }
                            _ => unreachable!(), // TODO
                        }
                    }
                    None => return Err(format!("Could not find column {} in sums", col)),
                };
            }
            QLNodeKind::Primitive(prim) => {
                let mut lhs: Option<&QLValue> = None;
                let mut rhs: Option<&QLValue> = None;
                for edge in graph.edges_directed(node_id, Direction::Incoming) {
                    let arg_id = *edge.weight();
                    if arg_id == 0 {
                        lhs = match &graph[edge.source()].value {
                            Some(v) => Some(&v),
                            None => None,
                        };
                    } else if arg_id == 1 {
                        rhs = match &graph[edge.source()].value {
                            Some(v) => Some(&v),
                            None => None,
                        };
                    }
                }
                let result = match prim {
                    QLPrimitive::Add => bin_broadcast_op!(lhs, rhs, |a, b| a + b),
                    QLPrimitive::Sub => bin_broadcast_op!(lhs, rhs, |a, b| a - b),
                    QLPrimitive::Mult => bin_broadcast_op!(lhs, rhs, |a, b| a * b),
                    QLPrimitive::Eq => cmp_broadcast_op!(lhs, rhs, |a, b| a == b),
                    QLPrimitive::Gt => cmp_broadcast_op!(lhs, rhs, |a, b| a > b),
                    QLPrimitive::Geq => cmp_broadcast_op!(lhs, rhs, |a, b| a >= b),
                    QLPrimitive::Lt => cmp_broadcast_op!(lhs, rhs, |a, b| a < b),
                    QLPrimitive::Leq => cmp_broadcast_op!(lhs, rhs, |a, b| a <= b),
                    QLPrimitive::And => bool_broadcast_op!(lhs, rhs, |a, b| a && b),
                    QLPrimitive::Or => bool_broadcast_op!(lhs, rhs, |a, b| a || b),
                    QLPrimitive::Not => match lhs {
                        Some(QLValue::Int(v)) => Ok(QLValue::Bool(*v == 0)),
                        Some(QLValue::Bool(v)) => Ok(QLValue::Bool(!*v)),
                        Some(QLValue::Float(v)) => Ok(QLValue::Bool(*v == 0.0)),
                        Some(QLValue::IntArray(v)) => Ok(QLValue::BoolArray(v.map(|e| *e == 0))),
                        Some(QLValue::BoolArray(v)) => Ok(QLValue::BoolArray(v.map(|e| !*e))),
                        Some(QLValue::FloatArray(v)) => {
                            Ok(QLValue::BoolArray(v.map(|e| *e == 0.0)))
                        }
                        None => unreachable!(),
                    },
                    QLPrimitive::Zeros => match lhs {
                        Some(QLValue::Int(size)) => Ok(QLValue::IntArray(Array1::<Int>::zeros(
                            (*size as usize,).f(),
                        ))),
                        _ => Err("cannot create non-int zeros array".to_string()),
                    },
                    QLPrimitive::Sum => match lhs {
                        Some(QLValue::IntArray(v)) => Ok(QLValue::Int(v.sum())),
                        Some(QLValue::BoolArray(v)) => Ok(QLValue::Int(v.map(|x| *x as Int).sum())),
                        Some(QLValue::FloatArray(v)) => Ok(QLValue::Float(v.sum())),
                        _ => Err("cannot sum non-array".to_string()),
                    },
                    QLPrimitive::Min => match lhs {
                        // TODO: better error handling here, e.g. for empty arrays.
                        Some(QLValue::IntArray(v)) => Ok(QLValue::Int(*v.min().unwrap())),
                        Some(QLValue::BoolArray(v)) => Ok(QLValue::Bool(*v.min().unwrap())),
                        Some(QLValue::FloatArray(v)) => Ok(QLValue::Float(*v.min_skipnan())),
                        _ => Err("cannot take min of non-array".to_string()),
                    },
                    QLPrimitive::Max => match lhs {
                        // TODO: better error handling here, e.g. for empty arrays.
                        Some(QLValue::IntArray(v)) => Ok(QLValue::Int(*v.max().unwrap())),
                        Some(QLValue::BoolArray(v)) => Ok(QLValue::Bool(*v.max().unwrap())),
                        Some(QLValue::FloatArray(v)) => Ok(QLValue::Float(*v.max_skipnan())),
                        _ => Err("cannot take max of non-array".to_string()),
                    },
                    QLPrimitive::Mean => match lhs {
                        // TODO: better error handling here, e.g. for empty arrays.
                        Some(QLValue::IntArray(v)) => {
                            Ok(QLValue::Float(v.map(|x| *x as Float).mean().unwrap()))
                        }
                        Some(QLValue::BoolArray(v)) => {
                            Ok(QLValue::Float(v.map(|x| bool_to_float(*x)).mean().unwrap()))
                        }
                        Some(QLValue::FloatArray(v)) => Ok(QLValue::Float(v.mean().unwrap())),
                        _ => Err("cannot take mean of non-array".to_string()),
                    },
                    QLPrimitive::Sort => match lhs {
                        // TODO: as of 2022-04-23, sorting is not available in
                        // the main rust-ndarray crate. However, it might be
                        // more efficient to eventually use the ndarray
                        // version where feasible--see
                        // https://github.com/rust-ndarray/ndarray/blob/
                        // 31244100631382bb8ee30721872a928bfdf07f44/examples/sort-axis.rs
                        Some(QLValue::IntArray(v)) => {
                            let mut v_sorted = v.to_vec();
                            v_sorted.sort();
                            Ok(QLValue::IntArray(Array1::from_vec(v_sorted)))
                        }
                        Some(QLValue::BoolArray(v)) => {
                            let mut v_sorted = v.to_vec();
                            v_sorted.sort();
                            Ok(QLValue::BoolArray(Array1::from_vec(v_sorted)))
                        }
                        Some(QLValue::FloatArray(v)) => {
                            let mut v_sorted = v.to_vec();
                            // TODO: handle NaN errors, etc. better...
                            v_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                            Ok(QLValue::FloatArray(Array1::from_vec(v_sorted)))
                        }
                        _ => Err("cannot sort non-array".to_string()),
                    },
                    /*
                    Divide,
                    Sort,
                    Mode,
                    */
                    _ => Err("not implemented yet!".to_string()),
                };
                graph[node_id].value = Some(result?);
            }
            _ => (),
        }
    }
    Ok(())
}

/// Computes the update order for a computation graph.
fn update_order(graph: &QLComputeGraph) -> Result<Vec<NodeIndex>, String> {
    let order = toposort(graph, None);
    if order.is_err() {
        return Err(format!("cycle in compute graph: {:?}", order.unwrap_err()));
    }
    Ok(order
        .unwrap()
        .into_iter()
        .filter(|nid| {
            match graph[*nid].kind {
                QLNodeKind::Column(_) => true,
                QLNodeKind::Primitive(_) => true,
                // Ignore constant nodes and the like.
                _ => false,
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_parse_expr_bool_literal_true() {
        assert_eq!(parse_expr("true".to_string()), Ok(QLExpr::Bool(true)))
    }

    #[test]
    fn test_parse_expr_bool_literal_false() {
        assert_eq!(parse_expr("false".to_string()), Ok(QLExpr::Bool(false)))
    }

    #[test]
    fn test_parse_expr_int_literal_positive() {
        assert_eq!(parse_expr("123".to_string()), Ok(QLExpr::Int(123)))
    }

    #[test]
    fn test_parse_expr_int_literal_negative() {
        assert_eq!(parse_expr("-123".to_string()), Ok(QLExpr::Int(-123)))
    }

    #[test]
    fn test_parse_expr_float_literal_positive() {
        assert_eq!(
            parse_expr("123.456".to_string()),
            Ok(QLExpr::Float(123.456))
        )
    }

    #[test]
    fn test_parse_expr_float_literal_negative() {
        assert_eq!(
            parse_expr("-123.456".to_string()),
            Ok(QLExpr::Float(-123.456))
        )
    }

    #[test]
    fn test_parse_expr_column_literal_valid() {
        assert_eq!(
            parse_expr(".TOTPOP".to_string()),
            Ok(QLExpr::Column(".TOTPOP".to_string()))
        )
    }

    #[test]
    fn test_parse_expr_var() {
        assert_eq!(
            parse_expr("abc".to_string()),
            Ok(QLExpr::Var("abc".to_string()))
        )
    }

    #[test]
    fn test_parse_expr_primitive() {
        assert_eq!(
            parse_expr("+".to_string()),
            Ok(QLExpr::Primitive(QLPrimitive::Add))
        )
    }

    #[test]
    fn test_parse_expr_add_two_ints() {
        assert_eq!(
            parse_expr("(+ 1 2)".to_string()),
            Ok(QLExpr::Apply(
                Box::new(QLExpr::Primitive(QLPrimitive::Add)),
                vec![QLExpr::Int(1), QLExpr::Int(2)]
            ))
        )
    }

    #[test]
    fn test_parse_expr_add_two_floats() {
        assert_eq!(
            parse_expr("(+ 1.0 2.0)".to_string()),
            Ok(QLExpr::Apply(
                Box::new(QLExpr::Primitive(QLPrimitive::Add)),
                vec![QLExpr::Float(1.0), QLExpr::Float(2.0)]
            ))
        )
    }

    #[test]
    fn test_parse_expr_add_two_columns() {
        assert_eq!(
            parse_expr("(+ .BPOP .TOTPOP)".to_string()),
            Ok(QLExpr::Apply(
                Box::new(QLExpr::Primitive(QLPrimitive::Add)),
                vec![
                    QLExpr::Column(".BPOP".to_string()),
                    QLExpr::Column(".TOTPOP".to_string())
                ]
            ))
        )
    }

    #[test]
    fn test_parse_expr_add_nested_expr() {
        assert_eq!(
            parse_expr("(+ (+ (+ .BPOP .TOTPOP) 1.0) 2)".to_string()),
            Ok(QLExpr::Apply(
                Box::new(QLExpr::Primitive(QLPrimitive::Add)),
                vec![
                    QLExpr::Apply(
                        Box::new(QLExpr::Primitive(QLPrimitive::Add)),
                        vec![
                            QLExpr::Apply(
                                Box::new(QLExpr::Primitive(QLPrimitive::Add)),
                                vec![
                                    QLExpr::Column(".BPOP".to_string()),
                                    QLExpr::Column(".TOTPOP".to_string())
                                ]
                            ),
                            QLExpr::Float(1.0)
                        ]
                    ),
                    QLExpr::Int(2)
                ]
            ))
        )
    }

    #[test]
    fn test_parse_expr_missing_open_paren() {
        assert_eq!(
            parse_expr("+ 1 2)".to_string()),
            // TODO: better error here?
            Err(QLError::ExtraChars),
        )
    }

    #[test]
    fn test_parse_expr_missing_close_paren() {
        assert_eq!(
            parse_expr("(+ 1 2".to_string()),
            Err(QLError::UnexpectedEOF),
        )
    }

    #[test]
    fn test_parse_expr_extra_close_paren() {
        assert_eq!(
            parse_expr("(+ 1 2))".to_string()),
            Err(QLError::UnmatchedParens),
        )
    }

    #[test]
    fn test_parse_expr_extra_close_parens() {
        assert_eq!(
            parse_expr("(+ 1 2)))".to_string()),
            Err(QLError::UnmatchedParens),
        )
    }

    #[test]
    fn test_parse_expr_extra_expr() {
        assert_eq!(
            parse_expr("(+ 1 2) (+ 1 2)".to_string()),
            Err(QLError::ExtraChars),
        )
    }

    #[test]
    fn test_parse_expr_applied_non_function() {
        assert_eq!(
            parse_expr("((+ 1 2))".to_string()),
            Err(QLError::AppliedNonFunction),
        )
    }
}
