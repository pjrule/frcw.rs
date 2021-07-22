/// ReCom chain benchmarks.
use std::time::Instant;
use std::default::Default;
use criterion::{criterion_group, criterion_main, Criterion};
use frcw::graph::Graph;
use frcw::partition::Partition;
use frcw::recom::run::multi_chain;
use frcw::recom::{RecomParams, RecomProposal, RecomVariant};
use frcw::stats::{ChainCounts, StatsWriter};
use test_fixtures::default_fixture;

/// RNG seed for all benchmarks.
const RNG_SEED: u64 = 153434375;

/// A `StatsWriter` that does nothing.
#[derive(Default)]
struct DummyWriter {}

impl StatsWriter for DummyWriter {
    fn init(&mut self, _graph: &Graph, _partition: &Partition) {}

    fn step(&mut self, _step: u64, _graph: &Graph, _proposal: &RecomProposal, _counts: &ChainCounts) {}

    fn close(&mut self) {}
}


fn grid_single_thread_recom_benchmark(c: &mut Criterion) {
    c.bench_function("ReCom, 6x6 grid, single-threaded", move |b| {
        b.iter_custom(|iters| {
            let (graph, partition) = default_fixture("6x6");
            let writer = Box::new(DummyWriter::default()) as Box<dyn StatsWriter>;
            let params = RecomParams {
                min_pop: 5,
                max_pop: 7,
                num_steps: iters,
                rng_seed: RNG_SEED,
                balance_ub: 0,
                variant: RecomVariant::DistrictPairs,
            };
            let start = Instant::now();
            multi_chain(&graph, &partition, writer, params, 1, 1);
            start.elapsed()
        })
    });
}


/*
fn ia_single_thread_recom_benchmark(c: &mut Criterion) {
}
*/

criterion_group!(benches, grid_single_thread_recom_benchmark);
criterion_main!(benches);
