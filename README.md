# [WIP] FRCW.rs

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
                      --n-steps 1000000 \
                      --n-threads 4 \
                      --pop-col TOTPOP \
                      --rng-seed 94915664 \
                      --tol 0.01 \
                      --batch-size 1 \ 
                      --variant cut_edges \
                      --sum-cols G16DPRS G16RPRS G16DHOR G16RHOR G18DSEN G18RSEN > va_recom.jsonl
```

(This takes ~5.5 seconds on my 2019 quad-core i5 MacBook Pro.)
