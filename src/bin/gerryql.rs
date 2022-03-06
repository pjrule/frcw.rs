//! Main CLI for GerryQL.
use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use clap::{value_t, App, Arg};
use frcw::init::graph_from_networkx;
use std::fs;
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
                .required(true)
                .help("The path of the dual graph (in NetworkX format)."),
        )
        .arg(
            Arg::with_name("chain_data")
                .long("chain-data")
                .takes_value(true)
                .required(true)
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
    let n_threads = value_t!(matches.value_of("n_threads"), usize).unwrap_or_else(|e| e.exit());
    let graph_json = fs::canonicalize(PathBuf::from(matches.value_of("graph_json").unwrap()))
        .unwrap()
        .into_os_string()
        .into_string()
        .unwrap();
    let pop_col = matches.value_of("pop_col").unwrap();

    let (graph, _) = graph_from_networkx(&graph_json, pop_col, vec![]).unwrap();
}

/// GerryQL basis functions.
#[derive(Copy, Clone, PartialEq, Debug)]
enum QLPrimitive {
    Add,
    Sub,
    Mult,
    Divide,
    Sum,
    Eq,
    Gt,
    Geq,
    Lt,
    Leq,
    And,
    Or,
    Not,
}

#[derive(Copy, Clone, PartialEq, Debug)]
enum QLPrimitiveKind {
    Unary,
    Binary,
    Cmp,
    Agg,
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

#[derive(Clone, PartialEq, Debug)]
enum QLError {
    UnexpectedEOF,
    MissingOpenParen,
    ExtraClosingParen,
    ExtraChars,
    ExpectedFuncName,
    UnreachableState,
    BadToken(String),
    TooFewArguments(QLPrimitive),
    TooManyArguments(QLPrimitive),
}

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
        "sum" => Some(QLPrimitive::Sum),
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
        QLPrimitive::Sum => QLPrimitiveKind::Agg,
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
        // We expect at least four tokens in the stream:
        // [open] [primitive name] [≥ 1 arg] [close]
        if tokens.len() < 4 {
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_parse_bool_literal_true() {
        assert_eq!(parse_expr("true".to_string()), Ok(QLExpr::Bool(true)));
    }

    #[test]
    fn test_parse_bool_literal_false() {
        assert_eq!(parse_expr("false".to_string()), Ok(QLExpr::Bool(false)));
    }

    #[test]
    fn test_parse_int_literal_positive() {
        assert_eq!(parse_expr("123".to_string()), Ok(QLExpr::Int(123)));
    }

    #[test]
    fn test_parse_int_literal_negative() {
        assert_eq!(parse_expr("-123".to_string()), Ok(QLExpr::Int(-123)));
    }

    #[test]
    fn test_parse_float_literal_positive() {
        assert_eq!(
            parse_expr("123.456".to_string()),
            Ok(QLExpr::Float(123.456))
        );
    }

    #[test]
    fn test_parse_float_literal_negative() {
        assert_eq!(
            parse_expr("-123.456".to_string()),
            Ok(QLExpr::Float(-123.456))
        );
    }
}
