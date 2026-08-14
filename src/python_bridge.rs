#![allow(unused_unsafe, clippy::unused_unsafe, unsafe_op_in_unsafe_fn)]

use crate::ICDFSampler16;
use pyo3::prelude::*;

#[pyclass(name = "ICDFSampler16")]
pub struct PyICDFSampler16(ICDFSampler16);

#[pymethods]
impl PyICDFSampler16 {
    #[new]
    pub fn new(x: Vec<f64>, cdf: Vec<f64>) -> Self {
        //let (sampler, _) = ICDFSampler16::from_symmetric_mean_relative_errors(&x, &cdf);
        //let (sampler, _) = ICDFSampler16::from_symmetric_max_relative_errors(&x, &cdf);
        let (sampler, _) =
            ICDFSampler16::new_with_error_fn(&x, &cdf, |f_interp, f_ref| (f_interp - f_ref).abs());

        Self(sampler)
    }

    pub fn sample(&self, probe: u16) -> f32 {
        self.0.sample(probe)
    }

    /// Sample the entire range of `u16` at once.
    pub fn sample_all(&self) -> Vec<f32> {
        (0..=u16::MAX).map(|p| self.0.sample(p)).collect()
    }
}

/// This function name MUST match the `name` in [lib] in Cargo.toml
#[pymodule]
fn icdf_sampler_16(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyICDFSampler16>()?;
    Ok(())
}
