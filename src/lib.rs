use pyo3::prelude::*;

/// Library definition for frcw. This is for usage by rust (such as src/bin files)
mod buffers;
pub mod config;
pub mod graph;
pub mod init;
pub mod partition;
pub mod recom;
mod spanning_tree;
pub mod stats;

/// This is collecting all of our wrapped functions to be exposed in the python module
#[pymodule]
fn frcw(py: Python, module: &PyModule) -> PyResult<()> {
    // add graph to the top-level module
    graph::init_submodule(module)?;

    Ok(())
}
