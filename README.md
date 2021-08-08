# frcw.rs (Fastest ReCom Chain in the West)

This is an ultra-high-performance implementation of the (reversible) [ReCom Markov chain for redistricting](https://arxiv.org/abs/1911.05725). It is intended to be used as a backend for [GerryChain.jl](https://github.com/mggg/GerryChainJulia/).

## Building
```sh
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

## Example usage

### Reversible ReCom
Running a 1,000,000-step reversible ReCom chain with [Virginia precinct data](https://github.com/mggg-states/VA-shapefiles):

```sh
./target/release/frcw --graph-json ./VA_precincts.json \
                      --assignment-col CD_16 \
                      --n-steps 1000000 \
                      --n-threads 8 \
                      --pop-col TOTPOP \
                      --rng-seed 94915664 \
                      --tol 0.01 \
                      --batch-size 64 \ 
                      --variant reversible \
                      --balance-ub 30 \
                      --sum-cols G16DPRS G16RPRS G16DHOR G16RHOR G18DSEN G18RSEN > va_revrecom.jsonl
```

(This takes ~7 seconds on my 2019 quad-core i5 MacBook Pro.)


### ReCom
Running a 100,000-step [GerryChain](https://github.com/mggg/gerrychain)-like ReCom chain with Virginia precinct data:
```sh
./target/release/frcw --graph-json ./VA_precincts.json \
                      --assignment-col CD_16 \
                      --n-steps 100000 \
                      --n-threads 4 \
                      --pop-col TOTPOP \
                      --rng-seed 94915664 \
                      --tol 0.01 \
                      --batch-size 1 \ 
                      --variant cut-edges-ust \
                      --sum-cols G16DPRS G16RPRS G16DHOR G16RHOR G18DSEN G18RSEN > va_recom.jsonl
```

(This takes ~5.5 seconds on my 2019 quad-core i5 MacBook Pro.)

## TODO
This project was originally a weekend project that lived in one `.rs` file, so it's a bit rough around the edges. The highest priorities are adding a bunch more tests and refactoring some particularly long functions.

- [x] Split into modules
- [x] Add docstrings
- [ ] Finish functional tests
  - [ ] Step-level invariants test _(in progress)_
    - [ ] Fix Crossbeam panic propagation in ReCom runner
    - [ ] Convert `multi_chain` to an iterator and separate writer out
  - [ ] Determinism test
  - [ ] Seed and freeze
  - [ ] RevReCom distribution tests (integrate Mai Nguyen's Google Summer of Code project)
- [ ] Add benchmarks _(in progress)_
- [ ] Add unit tests
- [ ] Add property tests (`quickcheck`) where appropriate
- [ ] Set up CI/CD (test and linting)
- [ ] Set up Codecov
- [ ] Refactoring 
  - [ ] Convert `RecomProposal` → `Proposal` and move to top level
  - [ ] Generalize fields in `Proposal` ({a, b} → `SmallVec`s)
  - [x] Generalize `ChainCounts` and remove count update ugliness in the ReCom runner
  - [x] Split up `stats` module
  - [ ] Rename sums → tallies for consistency with GerryChain
  - [ ] Define type aliases (i.e. don't hardcode `u32` everywhere)
    - [ ] Assess types: is using `u32` everywhere gaining us that much performance? What use cases might result in overflow?
  - [ ] Safe type coercion for input JSON
  - [ ] Sanity checks for input JSON (seed plan contiguity, seed plan population tolerance, etc.)
  - [ ] Break up long/confusing functions
    - [ ] `recom::run::multi_chain`
    - [ ] `recom::random_split` _(maybe)_
  - [ ] Enforce Rust idioms: remove `return` and `&Vec` where possible, etc.
  - [x] Make spanning tree statistics and other linear algebra-heavy features a crate-level feature?
  - [ ] Remove TSV writer? (in any case, should strongly encourage JSONL)
  - [ ] Struct marking which stats to collect?
  - [ ] `default` → `new` where appropriate
- [ ] New features (definite)
  - [ ] GerryChain-like scoring system for common use cases
    - [ ] Cut edge counts
    - [ ] Area & perimeter
    - [ ] Spanning tree statistics
    - [ ] ???
  - [ ] Make score calculations non-blocking (allow for multiple scoring threads?)
  - [ ] Batch size and thread count autotuning
  - [ ] Another round of performance optimizations
  - [ ] More ReCom variants
      - [ ] Add RMST sampling using Kruskal's algorithm
  - [x] Rectangular grid generator (useful for testing)
  - [ ] Minimal relabeling
  - [ ] Short bursts optimization (and general optimization framework)
- [ ] New features (possible)
  - [ ] Alternate input formats? (list of edges?)
  - [ ] Alternate output formats? (Parquet?)
  - [ ] Multi-member district support?
