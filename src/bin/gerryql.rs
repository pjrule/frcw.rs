//! Main CLI for GerryQL.
use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use clap::{value_t, App, Arg};
use frcw::init::graph_from_networkx;
use petgraph::dot::Dot;
use petgraph::graph::{Graph, NodeIndex};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;
use std::fs::{self, File};
use std::io::{self, stdin, BufRead, Write};
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

    loop {
        print!("> ");
        let _ = io::stdout().flush();
        let mut buffer = String::new();
        let _ = io::stdin().read_line(&mut buffer);
        let expr = parse_expr(buffer);
        match expr {
            Err(e) => println!("cannot parse expression: {:?}", e),
            //Ok(expr) => println!("expression: {}", expr)
            Ok(expr) => println!(
                "expression as compute graph: {}",
                Dot::new(&expr_to_graph(&expr))
            ),
        }
    }
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
        };
        write!(f, "{}", token)
    }
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
    UnmatchedParens,
    ExtraChars,
    UnreachableState,
    AppliedNonFunction,
    BadToken(String),
}

//// Parsed GerryQL tokens.
#[derive(Clone, PartialEq, Debug)]
enum QLToken {
    StartLambda,
    Start,
    End,
    BoolLiteral(bool),
    IntLiteral(i64),
    FloatLiteral(f64),
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

impl fmt::Display for QLComputeNode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.kind)
    }
}

/// Edges in the dependency graphs are labeled with argument IDs.
type QLComputeEdge = usize;

type QLComputeGraph = Graph<QLComputeNode, QLComputeEdge>;

/// Generates a computation graph from a GerryQL expression.
fn expr_to_graph(expr: &QLExpr) -> QLComputeGraph {
    let mut graph = QLComputeGraph::new();
    expr_to_node(expr, &mut graph, None, None);
    graph
}

/// Recursively builds a computation graph from a GerryQL expression.
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
}

/// Fills in all values in a computation graph.
/*
fn fill_graph(graph: &mut QLComputeGraph, column_sums: &BTreeMap<String, Vec<u32>>) {

}
*/

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

    fn test_parse_expr_missing_close_paren() {
        assert_eq!(
            parse_expr("(+ 1 2".to_string()),
            Err(QLError::UnexpectedEOF),
        )
    }

    fn test_parse_expr_extra_close_paren() {
        assert_eq!(
            parse_expr("(+ 1 2))".to_string()),
            Err(QLError::UnmatchedParens),
        )
    }

    fn test_parse_expr_extra_close_parens() {
        assert_eq!(
            parse_expr("(+ 1 2)))".to_string()),
            Err(QLError::ExtraChars),
        )
    }

    fn test_parse_expr_extra_expr() {
        assert_eq!(
            parse_expr("(+ 1 2) (+ 1 2)".to_string()),
            Err(QLError::ExtraChars),
        )
    }

    fn test_parse_expr_applied_non_function() {
        assert_eq!(
            parse_expr("((+ 1 2))".to_string()),
            Err(QLError::AppliedNonFunction),
        )
    }
}
