pub trait Kernel {
    fn sample_delay<R: rand::Rng + ?Sized>(&self, i: usize, j: usize, rng: &mut R) -> f64;

    fn validate_dimension(&self, _dimension: usize) -> Result<(), &'static str> {
        Ok(())
    }
}

use rand::distr::OpenClosed01;
use rand_distr::Distribution;

pub struct MixedExpKernel {
    pub(crate) lambda: Vec<Vec<Vec<f64>>>,
    pub(crate) cdf: Vec<Vec<Vec<f64>>>,
}

impl MixedExpKernel {
    pub fn new(
        mut weights: Vec<Vec<Vec<f64>>>,
        lambda: Vec<Vec<Vec<f64>>>,
    ) -> Result<Self, &'static str> {
        if weights.len() != lambda.len() {
            return Err("weights and beta must have the same first dimension");
        }
        for i in 0..weights.len() {
            if weights[i].len() != lambda[i].len() {
                return Err("weights and beta must have matching shapes");
            }
            for j in 0..weights[i].len() {
                if weights[i][j].len() != lambda[i][j].len() {
                    return Err("weights and beta must have matching shapes");
                }
                if weights[i][j].is_empty() {
                    return Err("mixture weights must be non-empty");
                }
                let mut total_weight = 0.0;
                for (k, &weight) in weights[i][j].iter().enumerate() {
                    if weight < 0.0 {
                        return Err("mixture weights must be non-negative");
                    }
                    let rate = lambda[i][j][k];
                    if rate <= 0.0 {
                        return Err("mixture rates must be positive");
                    }
                    total_weight += weight;
                }
                if total_weight <= 0.0 {
                    return Err("mixture weights must sum to a positive value");
                }
                let mut cumulative = 0.0;
                for weight in &mut weights[i][j] {
                    cumulative += *weight / total_weight;
                    *weight = cumulative;
                }
                if let Some(last) = weights[i][j].last_mut() {
                    *last = 1.0;
                }
            }
        }
        Ok(Self {
            lambda,
            cdf: weights,
        })
    }
}

pub struct ExpKernel {
    pub(crate) lambda: Vec<Vec<f64>>,
}

impl ExpKernel {
    pub fn new(lambda: Vec<Vec<f64>>) -> Self {
        Self { lambda }
    }
}

impl Kernel for ExpKernel {
    fn sample_delay<R: rand::Rng + ?Sized>(&self, i: usize, j: usize, rng: &mut R) -> f64 {
        let lambda = self.lambda[i][j];
        rand_distr::Exp::new(lambda).unwrap().sample(rng)
    }
}

pub struct LaggedExpKernel {
    pub(crate) lambda: Vec<Vec<f64>>,
    pub(crate) tau: Vec<Vec<f64>>,
}

impl LaggedExpKernel {
    pub fn new(lambda: Vec<Vec<f64>>, tau: Vec<Vec<f64>>) -> Result<Self, &'static str> {
        let d = lambda.len();
        if d == 0 || tau.len() != d {
            return Err("beta and tau must be non-empty square matrices with matching shapes");
        }
        for i in 0..d {
            if lambda[i].len() != d || tau[i].len() != d {
                return Err("beta and tau must be non-empty square matrices with matching shapes");
            }
            for j in 0..d {
                if !lambda[i][j].is_finite() || lambda[i][j] <= 0.0 {
                    return Err("beta must contain only finite positive values");
                }
                if !tau[i][j].is_finite() || tau[i][j] < 0.0 {
                    return Err("tau must contain only finite non-negative values");
                }
            }
        }
        Ok(Self { lambda, tau })
    }
}

impl Kernel for LaggedExpKernel {
    fn sample_delay<R: rand::Rng + ?Sized>(&self, i: usize, j: usize, rng: &mut R) -> f64 {
        let exponential_delay = rand_distr::Exp::new(self.lambda[i][j]).unwrap().sample(rng);
        self.tau[i][j] + exponential_delay
    }

    fn validate_dimension(&self, dimension: usize) -> Result<(), &'static str> {
        if self.lambda.len() != dimension {
            return Err("beta and tau dimensions must match the baseline dimension");
        }
        Ok(())
    }
}

pub struct GammaKernel {
    pub(crate) shape: Vec<Vec<f64>>,
    pub(crate) rate: Vec<Vec<f64>>,
}

impl GammaKernel {
    pub fn new(shape: Vec<Vec<f64>>, rate: Vec<Vec<f64>>) -> Self {
        Self { shape, rate }
    }
}

impl Kernel for GammaKernel {
    fn sample_delay<R: rand::Rng + ?Sized>(&self, i: usize, j: usize, rng: &mut R) -> f64 {
        let shape = self.shape[i][j];
        let rate = self.rate[i][j];
        rand_distr::Gamma::new(shape, 1.0 / rate)
            .unwrap()
            .sample(rng)
    }
}

impl Kernel for MixedExpKernel {
    fn sample_delay<R: rand::Rng + ?Sized>(&self, i: usize, j: usize, rng: &mut R) -> f64 {
        let cdf = &self.cdf[i][j];
        let u: f64 = rand::Rng::random(rng);
        let idx = cdf
            .iter()
            .position(|&threshold| u <= threshold)
            .unwrap_or(cdf.len() - 1);
        let lambda = self.lambda[i][j][idx];
        rand_distr::Exp::new(lambda).unwrap().sample(rng)
    }
}

pub struct PowerLawKernel {
    pub(crate) delta: Vec<Vec<f64>>,
    pub(crate) beta: Vec<Vec<f64>>,
}

impl PowerLawKernel {
    pub fn new(delta: Vec<Vec<f64>>, beta: Vec<Vec<f64>>) -> Result<Self, &'static str> {
        let d = delta.len();
        if d == 0 || beta.len() != d {
            return Err("delta and beta must be non-empty square matrices");
        }
        for i in 0..d {
            if delta[i].len() != d || beta[i].len() != d {
                return Err("delta and beta must be d×d");
            }
            for j in 0..d {
                if delta[i][j] <= 0.0 {
                    return Err("delta must be positive");
                }
                if beta[i][j] <= 1.0 {
                    return Err("beta must be greater than 1");
                }
            }
        }
        Ok(Self { delta, beta })
    }
}

impl Kernel for PowerLawKernel {
    fn sample_delay<R: rand::Rng + ?Sized>(&self, i: usize, j: usize, rng: &mut R) -> f64 {
        let u: f64 = rng.sample(OpenClosed01);
        let delta = self.delta[i][j];
        let beta = self.beta[i][j];
        let shape = beta - 1.0;
        delta * (u.powf(-1.0 / shape) - 1.0)
    }
}

pub enum KernelKind {
    Exponential(ExpKernel),
    LaggedExponential(LaggedExpKernel),
    Gamma(GammaKernel),
    MixedExponential(MixedExpKernel),
    PowerLaw(PowerLawKernel),
}

impl Kernel for KernelKind {
    fn sample_delay<R: rand::Rng + ?Sized>(&self, i: usize, j: usize, rng: &mut R) -> f64 {
        match self {
            Self::Exponential(kernel) => kernel.sample_delay(i, j, rng),
            Self::LaggedExponential(kernel) => kernel.sample_delay(i, j, rng),
            Self::Gamma(kernel) => kernel.sample_delay(i, j, rng),
            Self::MixedExponential(kernel) => kernel.sample_delay(i, j, rng),
            Self::PowerLaw(kernel) => kernel.sample_delay(i, j, rng),
        }
    }

    fn validate_dimension(&self, dimension: usize) -> Result<(), &'static str> {
        match self {
            Self::Exponential(kernel) => kernel.validate_dimension(dimension),
            Self::LaggedExponential(kernel) => kernel.validate_dimension(dimension),
            Self::Gamma(kernel) => kernel.validate_dimension(dimension),
            Self::MixedExponential(kernel) => kernel.validate_dimension(dimension),
            Self::PowerLaw(kernel) => kernel.validate_dimension(dimension),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Kernel, LaggedExpKernel};
    use rand::SeedableRng;

    #[test]
    fn lagged_exponential_delay_has_expected_support_and_mean() {
        let kernel = LaggedExpKernel::new(vec![vec![2.0]], vec![vec![0.4]]).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(20260901);
        let sample_count = 50_000;
        let mut total = 0.0;

        for _ in 0..sample_count {
            let delay = kernel.sample_delay(0, 0, &mut rng);
            assert!(delay > 0.4);
            total += delay;
        }

        let sample_mean = total / sample_count as f64;
        let expected_mean = 0.4 + 1.0 / 2.0;
        assert!((sample_mean - expected_mean).abs() < 0.02);
    }
}
