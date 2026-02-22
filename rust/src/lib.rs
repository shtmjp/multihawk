use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod baseline;
use baseline::{BaselineConst, BaselineKind, BaselinePiecewiseConst, BaselinePiecewiseLinear};
mod pp_data;
use pp_data::MultivariatePPData;
mod simulate;
use simulate::{simulate_hawkes_branching_with_baseline, ImmigrantSamplingMethod};
mod kernel;
use kernel::{ExpKernel, GammaKernel, KernelKind, MixedExpKernel, PowerLawKernel};

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_pcg::{Pcg64, Pcg64Dxsm};
use serde::{Deserialize, Serialize};

use pyo3::types::PyDict;
use pyo3::{Bound, PyObject, Python};

#[derive(Serialize, Deserialize)]
struct Pcg128State {
    state: u128,
    increment: u128,
}

fn pcg64_parts(rng: &Pcg64) -> (u128, u128) {
    let bytes = bincode::serialize(rng).expect("serialize Pcg64");
    let snap: Pcg128State = bincode::deserialize(&bytes).expect("decode Pcg64 -> Pcg128State");
    (snap.state, snap.increment)
}

fn pcg64dxsm_parts(rng: &Pcg64Dxsm) -> (u128, u128) {
    let bytes = bincode::serialize(rng).expect("serialize Pcg64Dxsm");
    let snap: Pcg128State = bincode::deserialize(&bytes).expect("decode Pcg64Dxsm -> Pcg128State");
    (snap.state, snap.increment)
}

fn pcg64_from_parts(state: u128, increment: u128) -> Pcg64 {
    let snap = Pcg128State { state, increment };
    let bytes = bincode::serialize(&snap).expect("encode snapshot");
    bincode::deserialize(&bytes).expect("decode snapshot -> Pcg64")
}

fn pcg64dxsm_from_parts(state: u128, increment: u128) -> Pcg64Dxsm {
    let snap = Pcg128State { state, increment };
    let bytes = bincode::serialize(&snap).expect("encode snapshot");
    bincode::deserialize(&bytes).expect("decode snapshot -> Pcg64Dxsm")
}

fn parse_sampling_method(method: &str) -> PyResult<ImmigrantSamplingMethod> {
    match method {
        "thinning" => Ok(ImmigrantSamplingMethod::Thinning),
        "inverse_transform" => Ok(ImmigrantSamplingMethod::InverseTransform),
        other => Err(PyValueError::new_err(format!(
            "unsupported immigrant sampling method '{other}'"
        ))),
    }
}

#[pyclass]
pub struct PyMultivariatePPData {
    #[pyo3(get)]
    pub timestamps: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub obs_window: (f64, f64),
}

impl From<MultivariatePPData> for PyMultivariatePPData {
    fn from(data: MultivariatePPData) -> Self {
        Self {
            timestamps: data.timestamps,
            obs_window: data.obs_window,
        }
    }
}

fn baseline_from_object(py: Python<'_>, obj: PyObject) -> PyResult<BaselineKind> {
    let dict = obj.downcast_bound::<PyDict>(py)?;
    let kind_value = dict
        .get_item("kind")?
        .ok_or_else(|| PyValueError::new_err("baseline spec must include 'kind'"))?;
    let params_value = dict
        .get_item("params")?
        .ok_or_else(|| PyValueError::new_err("baseline spec must include 'params'"))?;
    let kind: String = kind_value.extract()?;
    let params = params_value.downcast::<PyDict>()?;
    match kind.as_str() {
        "constant" => {
            let values: Vec<f64> = params
                .get_item("values")?
                .ok_or_else(|| PyValueError::new_err("baseline params must include 'values'"))?
                .extract()?;
            BaselineConst::new(values)
                .map(BaselineKind::Constant)
                .map_err(PyValueError::new_err)
        }
        "piecewise_constant" => {
            let breaks: Vec<f64> = params
                .get_item("breaks")?
                .ok_or_else(|| PyValueError::new_err("baseline params must include 'breaks'"))?
                .extract()?;
            let rates: Vec<Vec<f64>> = params
                .get_item("rates")?
                .ok_or_else(|| PyValueError::new_err("baseline params must include 'rates'"))?
                .extract()?;
            BaselinePiecewiseConst::new(breaks, rates)
                .map(BaselineKind::PiecewiseConst)
                .map_err(PyValueError::new_err)
        }
        "piecewise_linear" => {
            let breaks: Vec<f64> = params
                .get_item("breaks")?
                .ok_or_else(|| PyValueError::new_err("baseline params must include 'breaks'"))?
                .extract()?;
            let values: Vec<Vec<f64>> = params
                .get_item("values")?
                .ok_or_else(|| PyValueError::new_err("baseline params must include 'values'"))?
                .extract()?;
            BaselinePiecewiseLinear::new(breaks, values)
                .map(BaselineKind::PiecewiseLinear)
                .map_err(PyValueError::new_err)
        }
        other => Err(PyValueError::new_err(format!(
            "unsupported baseline kind '{other}'"
        ))),
    }
}

fn kernel_from_object(py: Python<'_>, obj: PyObject) -> PyResult<KernelKind> {
    let dict = obj.downcast_bound::<PyDict>(py)?;
    let kind_value = dict
        .get_item("kind")?
        .ok_or_else(|| PyValueError::new_err("kernel spec must include 'kind'"))?;
    let params_value = dict
        .get_item("params")?
        .ok_or_else(|| PyValueError::new_err("kernel spec must include 'params'"))?;
    let kind: String = kind_value.extract()?;
    let params = params_value.downcast::<PyDict>()?;
    match kind.as_str() {
        "exponential" => {
            let beta: Vec<Vec<f64>> = params
                .get_item("beta")?
                .ok_or_else(|| PyValueError::new_err("kernel params must include 'beta'"))?
                .extract()?;
            Ok(KernelKind::Exponential(ExpKernel::new(beta)))
        }
        "gamma" => {
            let shape: Vec<Vec<f64>> = params
                .get_item("shape")?
                .ok_or_else(|| PyValueError::new_err("kernel params must include 'shape'"))?
                .extract()?;
            let rate: Vec<Vec<f64>> = params
                .get_item("rate")?
                .ok_or_else(|| PyValueError::new_err("kernel params must include 'rate'"))?
                .extract()?;
            Ok(KernelKind::Gamma(GammaKernel::new(shape, rate)))
        }
        "mixed_exponential" => {
            let weights: Vec<Vec<Vec<f64>>> = params
                .get_item("weights")?
                .ok_or_else(|| PyValueError::new_err("kernel params must include 'weights'"))?
                .extract()?;
            let beta: Vec<Vec<Vec<f64>>> = params
                .get_item("beta")?
                .ok_or_else(|| PyValueError::new_err("kernel params must include 'beta'"))?
                .extract()?;
            MixedExpKernel::new(weights, beta)
                .map(KernelKind::MixedExponential)
                .map_err(PyValueError::new_err)
        }
        "power_law" => {
            let delta_value = params.get_item("delta")?;
            let delta_item = if let Some(value) = delta_value {
                Some(value)
            } else {
                params.get_item("cutoff")?
            };
            let delta: Vec<Vec<f64>> = delta_item
                .ok_or_else(|| {
                    PyValueError::new_err("kernel params must include 'delta' or 'cutoff'")
                })?
                .extract()?;
            let beta_value = params.get_item("beta")?;
            let beta_item = if let Some(value) = beta_value {
                Some(value)
            } else {
                params.get_item("exponent")?
            };
            let beta: Vec<Vec<f64>> = beta_item
                .ok_or_else(|| {
                    PyValueError::new_err("kernel params must include 'beta' or 'exponent'")
                })?
                .extract()?;
            PowerLawKernel::new(delta, beta)
                .map(KernelKind::PowerLaw)
                .map_err(PyValueError::new_err)
        }
        other => Err(PyValueError::new_err(format!(
            "unsupported kernel kind '{other}'"
        ))),
    }
}

fn simulate_with_rng<R: rand::Rng + ?Sized>(
    t_max: f64,
    baseline: &BaselineKind,
    alpha: &[Vec<f64>],
    kernel: &KernelKind,
    method: ImmigrantSamplingMethod,
    rng: &mut R,
) -> PyResult<PyMultivariatePPData> {
    simulate_hawkes_branching_with_baseline(t_max, baseline, alpha, kernel, method, rng)
        .map(Into::into)
        .map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (t_max, baseline, alpha, kernel, seed=None, sampling_method="thinning"))]
fn simulate_hawkes(
    py: Python<'_>,
    t_max: f64,
    baseline: PyObject,
    alpha: Vec<Vec<f64>>,
    kernel: PyObject,
    seed: Option<u64>,
    sampling_method: &str,
) -> PyResult<PyMultivariatePPData> {
    let baseline = baseline_from_object(py, baseline)?;
    let kernel = kernel_from_object(py, kernel)?;
    let method = parse_sampling_method(sampling_method)?;
    let mut rng: StdRng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_os_rng(),
    };
    simulate_with_rng(t_max, &baseline, &alpha, &kernel, method, &mut rng)
}

#[pyfunction]
#[pyo3(signature = (t_max, baseline, alpha, kernel, state, inc, sampling_method="thinning"))]
fn simulate_hawkes_pcg64(
    py: Python<'_>,
    t_max: f64,
    baseline: PyObject,
    alpha: Vec<Vec<f64>>,
    kernel: PyObject,
    state: u128,
    inc: u128,
    sampling_method: &str,
) -> PyResult<(PyMultivariatePPData, u128, u128)> {
    let baseline = baseline_from_object(py, baseline)?;
    let kernel = kernel_from_object(py, kernel)?;
    let method = parse_sampling_method(sampling_method)?;
    let mut rng = pcg64_from_parts(state, inc);
    simulate_with_rng(t_max, &baseline, &alpha, &kernel, method, &mut rng).map(|data| {
        let (s, i) = pcg64_parts(&rng);
        (data, s, i)
    })
}

#[pyfunction]
#[pyo3(signature = (t_max, baseline, alpha, kernel, state, inc, sampling_method="thinning"))]
fn simulate_hawkes_pcg64dxsm(
    py: Python<'_>,
    t_max: f64,
    baseline: PyObject,
    alpha: Vec<Vec<f64>>,
    kernel: PyObject,
    state: u128,
    inc: u128,
    sampling_method: &str,
) -> PyResult<(PyMultivariatePPData, u128, u128)> {
    let baseline = baseline_from_object(py, baseline)?;
    let kernel = kernel_from_object(py, kernel)?;
    let method = parse_sampling_method(sampling_method)?;
    let mut rng = pcg64dxsm_from_parts(state, inc);
    simulate_with_rng(t_max, &baseline, &alpha, &kernel, method, &mut rng).map(|data| {
        let (s, i) = pcg64dxsm_parts(&rng);
        (data, s, i)
    })
}

#[pymodule]
fn multihawk_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMultivariatePPData>()?;
    m.add_function(wrap_pyfunction!(simulate_hawkes, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_hawkes_pcg64, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_hawkes_pcg64dxsm, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngCore;

    #[test]
    fn resume_pcg64_exact() {
        let mut a = Pcg64::from_os_rng();
        let _ = a.next_u64();
        let (state, increment) = pcg64_parts(&a);

        let mut b = pcg64_from_parts(state, increment);
        assert_eq!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn resume_pcg64dxsm_exact() {
        let mut a = Pcg64Dxsm::from_os_rng();
        let _ = a.next_u64();
        let (state, increment) = pcg64dxsm_parts(&a);

        let mut b = pcg64dxsm_from_parts(state, increment);
        assert_eq!(a.next_u64(), b.next_u64());
    }
}
