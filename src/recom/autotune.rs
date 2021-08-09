//! Batch size autotuning for ReCom chains.

struct BatchSizeTuner {
    /// Minimum per-thread batch size.
    min_size: usize,
    /// Maximum per-thread batch size.
    max_size: usize,
}

impl BatchSizeTuner {
    pub fn new(min_size: usize, max_size: usize) -> BatchSizeTuner {
        BatchSizeTuner {
            min_size: min_size,
            max_size: max_size,
        }
    }
}
