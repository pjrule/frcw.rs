//! Chain self-loop statistics.
use serde::ser::{Serialize, SerializeStruct, Serializer};
use std::collections::HashMap;
use std::ops::Add;

/// Reasons why a self-loop occurred in a Markov chain.
#[derive(Hash, Eq, PartialEq, Copy, Clone)]
pub enum SelfLoopReason {
    /// Drew non-adjacent district pairs.
    NonAdjacent,
    /// Drew a spanning tree with no ε-balance nodes
    /// (and therefore no valid splits).
    NoSplit,
    /// Probabilistic rejection based on seam length
    /// (reversible ReCom only).
    SeamLength,
}

/// Self-loop statistics since the last accepted proposal.
pub struct SelfLoopCounts {
    counts: HashMap<SelfLoopReason, usize>,
}

impl Default for SelfLoopCounts {
    fn default() -> SelfLoopCounts {
        return SelfLoopCounts {
            counts: HashMap::new(),
        };
    }
}

impl Add for SelfLoopCounts {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut union = self.counts.clone();
        for (&reason, count) in other.counts.iter() {
            *union.entry(reason).or_insert(0) += count;
        }
        return SelfLoopCounts { counts: union };
    }
}

impl SelfLoopCounts {
    /// Increments the self-loop count (with a reason).
    pub fn inc(&mut self, reason: SelfLoopReason) {
        *self.counts.entry(reason).or_insert(0) += 1;
    }

    /// Decrements the self-loop count (with a reason).
    pub fn dec(&mut self, reason: SelfLoopReason) {
        *self.counts.entry(reason).or_insert(0) -= 1;
    }

    /// Returns the self-loop count for a reason.
    pub fn get(&self, reason: SelfLoopReason) -> usize {
        self.counts.get(&reason).map_or(0, |&c| c)
    }

    /// Returns the total self-loop count over all reasons.
    pub fn sum(&self) -> usize {
        self.counts.iter().map(|(_, c)| c).sum()
    }

    /// Retrieves an event from the counts based on an arbitrary ordering
    /// and removes the event from the counts.
    ///
    /// Used for sampling events in multithreaded chains: because `SelfLoopCounts`
    /// doesn't store accepted proposal counts, we often want to draw a random event,
    /// accept a proposal if the event index is over/under a threshold, and self-loop
    /// otherwise.
    pub fn index_and_dec(&mut self, index: usize) -> Option<SelfLoopReason> {
        let mut seen = 0;
        for (&reason, count) in self.counts.iter() {
            if seen <= index && index < seen + count {
                self.dec(reason);
                return Some(reason);
            }
            seen += count;
        }
        None
    }
}

impl Serialize for SelfLoopCounts {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("SelfLoopCounts", self.counts.len())?;
        for (&reason, count) in self.counts.iter() {
            // Use camel-case field names in Serde serialization.
            let key = match reason {
                SelfLoopReason::NonAdjacent => "non_adjacent",
                SelfLoopReason::NoSplit => "no_split",
                SelfLoopReason::SeamLength => "seam_length",
            };
            state.serialize_field(key, count)?;
        }
        state.end()
    }
}
