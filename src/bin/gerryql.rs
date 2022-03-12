//! Main CLI for GerryQL.
use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use clap::{value_t, App, Arg};
use frcw::init::graph_from_networkx;
use petgraph::graph::{Graph, NodeIndex};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufRead};
use std::path::{Path, PathBuf};

fn main() {
    let cli = App::new("gerryql")
        .version("0.1.0")
        .author("Parker J. Rule <parker.rule@tufts.edu>")
        .about("A query engine for statistics on long Markov chain runs.")
        .arg(
            Arg::with_name("graph_json")
                .long("graph-json")
                .takes_value(true)
                //.required(true)
                .help("The path of the dual graph (in NetworkX format)."),
        )
        .arg(
            Arg::with_name("chain_data")
                .long("chain-data")
                .takes_value(true)
                //.required(true)
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
    /*
    let n_threads = value_t!(matches.value_of("n_threads"), usize).unwrap_or_else(|e| e.exit());
    let graph_json = fs::canonicalize(PathBuf::from(matches.value_of("graph_json").unwrap()))
        .unwrap()
        .into_os_string()
        .into_string()
        .unwrap();
    let chain_data_path =
        fs::canonicalize(PathBuf::from(matches.value_of("chain_data").unwrap())).unwrap();
    */
    println!("val,freq");
    for (val, freq) in stats_demo().unwrap().iter() {
        println!("{},{}", val, freq);
    }

    //let (graph, _) = graph_from_networkx(&graph_json, pop_col, vec![]).unwrap();
}

/// Hardcoded stats from stdin (will be replaced later).
fn stats_demo() -> Result<HashMap<u64, u64>, io::Error> {
    let bpop_col = "BPOP20";
    let mut started = false;
    let mut black_shares = Vec::<f64>::new();
    let mut black_maj_hist = HashMap::<u64, u64>::new();
    for (line, contents) in io::stdin().lock().lines().enumerate() {
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
            let pop_data = init_data.as_object().unwrap()["populations"]
                .as_array()
                .unwrap();
            let bpop_data = init_data.as_object().unwrap()["sums"].as_object().unwrap()[bpop_col]
                .as_array()
                .unwrap();
            let pops: Vec<u64> = pop_data.iter().map(|c| c.as_u64().unwrap()).collect();
            let bpops: Vec<u64> = bpop_data.iter().map(|c| c.as_u64().unwrap()).collect();
            black_shares = pops
                .iter()
                .zip(bpops.iter())
                .map(|(p, bp)| *bp as f64 / *p as f64)
                .collect();
            started = true;
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
            let step_pops: Vec<u64> = step_data["populations"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| d.as_u64().unwrap())
                .collect();
            let step_bpops: Vec<u64> = step_data["sums"].as_object().unwrap()[bpop_col]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| c.as_u64().unwrap())
                .collect();
            for ((dist, pop), bpop) in dists.iter().zip(step_pops.iter()).zip(step_bpops.iter()) {
                black_shares[*dist] = *bpop as f64 / *pop as f64;
            }
            let black_maj_count = black_shares.iter().filter(|s| **s >= 0.5).count() as u64;
            let bin_count = match black_maj_hist.get(&black_maj_count) {
                Some(c) => *c,
                None => 0,
            };
            black_maj_hist.insert(black_maj_count, bin_count + 1);
        }
    }
    Ok(black_maj_hist)
}

/// GerryQL basis functions.
#[derive(Copy, Clone, PartialEq, Debug)]
enum QLPrimitive {
    Add,
    Sub,
    Mult,
    Divide,
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
}

/// Kinds of primitive GerryQL
#[derive(Copy, Clone, PartialEq, Debug)]
enum QLPrimitiveKind {
    Unary,
    Binary,
    Cmp,
    Agg,
    Bool,
}

/// GerryQL expressions.
#[derive(Clone, PartialEq, Debug)]
enum QLExpr {
    Bool(bool),
    Int(i64),
    Float(f64),
    PrimitiveCall(QLPrimitive, Vec<QLExpr>),
    Column(String),
}

/// GerryQL values.
#[derive(Clone, PartialEq, Debug)]
enum QLValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    VecBool(Vec<bool>),
    VecInt(Vec<i64>),
    VecFloat(Vec<f64>),
}

/// GerryQL errors.
#[derive(Clone, PartialEq, Debug)]
enum QLError {
    UnexpectedEOF,
    MissingOpenParen,
    ExtraCloseParen,
    ExtraChars,
    ExpectedFuncName,
    UnreachableState,
    BadToken(String),
    TooFewArguments(QLPrimitive),
    TooManyArguments(QLPrimitive),
}

//// Parsed GerryQL tokens.
#[derive(Clone, PartialEq, Debug)]
enum QLToken {
    StartCall,
    EndCall,
    BoolLiteral(bool),
    IntLiteral(i64),
    FloatLiteral(f64),
    ColumnLiteral(String),
    PrimitiveName(QLPrimitive),
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
        _ => None,
    }
}

/// Converts a token to a primitive function identifier, if possible.
fn primitive_kind(prim: QLPrimitive) -> QLPrimitiveKind {
    match prim {
        QLPrimitive::Add => QLPrimitiveKind::Binary,
        QLPrimitive::Sub => QLPrimitiveKind::Binary,
        QLPrimitive::Divide => QLPrimitiveKind::Binary,
        QLPrimitive::Mult => QLPrimitiveKind::Binary,
        QLPrimitive::Sort => QLPrimitiveKind::Unary,
        QLPrimitive::Sum => QLPrimitiveKind::Agg,
        QLPrimitive::Min => QLPrimitiveKind::Agg,
        QLPrimitive::Max => QLPrimitiveKind::Agg,
        QLPrimitive::Median => QLPrimitiveKind::Agg,
        QLPrimitive::Mode => QLPrimitiveKind::Agg,
        QLPrimitive::Mean => QLPrimitiveKind::Agg,
        QLPrimitive::Eq => QLPrimitiveKind::Cmp,
        QLPrimitive::Gt => QLPrimitiveKind::Cmp,
        QLPrimitive::Geq => QLPrimitiveKind::Cmp,
        QLPrimitive::Lt => QLPrimitiveKind::Cmp,
        QLPrimitive::Leq => QLPrimitiveKind::Cmp,
        QLPrimitive::And => QLPrimitiveKind::Binary,
        QLPrimitive::Or => QLPrimitiveKind::Binary,
        QLPrimitive::Not => QLPrimitiveKind::Unary,
    }
}

fn parse_token(token: &str) -> QLToken {
    if let Some(prim) = token_to_primitive(token) {
        return QLToken::PrimitiveName(prim);
    }
    if let Ok(val) = token.parse::<i64>() {
        return QLToken::IntLiteral(val);
    }
    if let Ok(val) = token.parse::<f64>() {
        return QLToken::FloatLiteral(val);
    }
    if token.len() > 1 {
        if let Some('.') = token.chars().next() {
            return QLToken::ColumnLiteral(token.to_owned());
        }
    }
    match token {
        "(" => QLToken::StartCall,
        "[" => QLToken::StartCall,
        ")" => QLToken::EndCall,
        "]" => QLToken::EndCall,
        "true" => QLToken::BoolLiteral(true),
        "false" => QLToken::BoolLiteral(false),
        _ => QLToken::Unknown,
    }
}

fn parse_next_expr(tokens: &[String]) -> Result<(QLExpr, &[String]), QLError> {
    if tokens.len() == 0 {
        return Err(QLError::UnexpectedEOF);
    }
    let token = &tokens[0];
    let tt = parse_token(&token);
    if tt == QLToken::StartCall {
        // We expect at least three tokens in the stream:
        // [open] [primitive name] [close]
        if tokens.len() < 3 {
            return Err(QLError::UnexpectedEOF);
        }
        if let Some(prim) = token_to_primitive(&tokens[1]) {
            // Each primitive takes at least one argument; some
            // may take two.
            let mut args: Vec<QLExpr> = vec![];
            let (arg1, mut rest) = match parse_next_expr(&tokens[2..]) {
                Ok(res) => res,
                Err(QLError::UnexpectedEOF) => return Err(QLError::TooFewArguments(prim)),
                Err(e) => return Err(e),
            };
            args.push(arg1);
            if primitive_kind(prim) == QLPrimitiveKind::Binary
                || primitive_kind(prim) == QLPrimitiveKind::Cmp
            {
                let (arg2, rest2) = match parse_next_expr(rest) {
                    Ok(res) => res,
                    Err(QLError::UnexpectedEOF) => return Err(QLError::TooFewArguments(prim)),
                    Err(e) => return Err(e),
                };
                args.push(arg2);
                rest = rest2;
            }
            if rest.len() > 0 && parse_token(&rest[0]) != QLToken::EndCall {
                return Err(QLError::TooManyArguments(prim));
            }
            if rest.len() == 0 {
                return Err(QLError::UnexpectedEOF);
            }
            return Ok((QLExpr::PrimitiveCall(prim, args), &rest[1..]));
        } else {
            return Err(QLError::ExpectedFuncName);
        }
    }

    let rest = &tokens[1..];
    match tt {
        QLToken::BoolLiteral(val) => Ok((QLExpr::Bool(val), rest)),
        QLToken::FloatLiteral(val) => Ok((QLExpr::Float(val), rest)),
        QLToken::IntLiteral(val) => Ok((QLExpr::Int(val), rest)),
        QLToken::ColumnLiteral(val) => Ok((QLExpr::Column(val), rest)),
        QLToken::EndCall => Err(QLError::UnexpectedEOF),
        QLToken::Unknown => Err(QLError::BadToken(token.to_owned())),
        QLToken::PrimitiveName(_) => Err(QLError::MissingOpenParen),
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
                Err(QLError::ExtraChars)
            } else {
                Ok(exp)
            }
        }
    }
}

/// What kind of computation does the data flow graph represent?
enum QLNodeKind {
    Primitive(QLPrimitive),
    Constant,
    Column(String),
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
    kind: QLNodeKind,
    update: QLUpdate,
    value: Option<QLValue>,
}

/// Edges in the dependency graphs are labeled with argument IDs.
type QLComputeEdge = usize;

type QLComputeGraph = Graph<QLComputeNode, QLComputeEdge>;

/// Generates a computation graph from a GerryQL expression.
fn expr_to_node(
    expr: &QLExpr,
    graph: &mut QLComputeGraph,
    parent: Option<NodeIndex>,
    child_id: Option<QLComputeEdge>,
) {
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
        QLExpr::PrimitiveCall(prim, args) => {
            (QLNodeKind::Primitive(prim.to_owned()), None, Some(args))
        }
        QLExpr::Column(col) => (QLNodeKind::Column(col.to_owned()), None, None),
    };
    let node_idx = graph.add_node(QLComputeNode {
        kind: kind,
        value: value,
        update: QLUpdate::All, // Be conservative initially.
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

    //
}

// _column_sums: &HashMap<String, Vec<u32>>

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
    fn test_parse_expr_column_literal_invalid() {
        assert_eq!(
            parse_expr(".".to_string()), // can't have empty column name
            Err(QLError::BadToken(".".to_string()))
        )
    }

    #[test]
    fn test_parse_expr_add_two_ints() {
        assert_eq!(
            parse_expr("(+ 1 2)".to_string()),
            Ok(QLExpr::PrimitiveCall(
                QLPrimitive::Add,
                vec![QLExpr::Int(1), QLExpr::Int(2)]
            ))
        )
    }

    #[test]
    fn test_parse_expr_add_two_floats() {
        assert_eq!(
            parse_expr("(+ 1.0 2.0)".to_string()),
            Ok(QLExpr::PrimitiveCall(
                QLPrimitive::Add,
                vec![QLExpr::Float(1.0), QLExpr::Float(2.0)]
            ))
        )
    }

    #[test]
    fn test_parse_expr_add_two_columns() {
        assert_eq!(
            parse_expr("(+ .BPOP .TOTPOP)".to_string()),
            Ok(QLExpr::PrimitiveCall(
                QLPrimitive::Add,
                vec![
                    QLExpr::Column(".BPOP".to_string()),
                    QLExpr::Column(".TOTPOP".to_string())
                ]
            ))
        )
    }

    #[test]
    fn test_parse_expr_add_one_arg() {
        assert_eq!(
            parse_expr("(+ 1)".to_string()),
            Err(QLError::TooFewArguments(QLPrimitive::Add))
        )
    }

    #[test]
    fn test_parse_expr_add_three_arg() {
        assert_eq!(
            parse_expr("(+ 1 2 3)".to_string()),
            Err(QLError::TooManyArguments(QLPrimitive::Add))
        )
    }

    #[test]
    fn test_parse_expr_add_nested_expr() {
        assert_eq!(
            parse_expr("(+ (+ (+ .BPOP .TOTPOP) 1.0) 2)".to_string()),
            Ok(QLExpr::PrimitiveCall(
                QLPrimitive::Add,
                vec![
                    QLExpr::PrimitiveCall(
                        QLPrimitive::Add,
                        vec![
                            QLExpr::PrimitiveCall(
                                QLPrimitive::Add,
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
    fn test_parse_expr_unknown_function_name() {
        assert_eq!(
            parse_expr("(1)".to_string()),
            Err(QLError::ExpectedFuncName),
        )
    }

    #[test]
    fn test_parse_expr_missing_open_paren() {
        assert_eq!(
            parse_expr("+ 1 2)".to_string()),
            Err(QLError::MissingOpenParen),
        )
    }

    fn test_parse_expr_missing_close_paren() {
        assert_eq!(
            parse_expr("(+ 1 2".to_string()),
            Err(QLError::UnexpectedEOF),
        )
    }

    fn test_parse_expr_extra_close_paren() {
        assert_eq!(
            parse_expr("(+ 1 2))".to_string()),
            Err(QLError::ExtraCloseParen),
        )
    }

    fn test_parse_expr_extra_close_parens() {
        assert_eq!(
            parse_expr("(+ 1 2)))".to_string()),
            Err(QLError::ExtraCloseParen),
        )
    }
}
