//! Helpers for parsing JSON configuration strings.
use serde_json::{from_str, Value};
use std::collections::HashMap;

pub fn parse_region_weights_config(region_weights_raw: &str) -> Option<Vec<(String, f64)>> {
    match region_weights_raw {
        "" => None,
        raw => {
            let mut weights: Vec<(String, f64)> = from_str::<HashMap<&str, Value>>(raw)
                .unwrap()
                .into_iter()
                .map(|(k, v)| (k.to_owned(), v.as_f64().unwrap()))
                .collect();
            // Sort region weights in descending order (highest priority -> lowest priority).
            weights.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().reverse());
            Some(weights)
        }
    }
}
