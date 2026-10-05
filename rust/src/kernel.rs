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
        let d = weights.len();
        if d == 0 || lambda.len() != d {
            return Err("weights and beta must be non-empty square tensors with matching shapes");
        }
        for i in 0..d {
            if weights[i].len() != d || lambda[i].len() != d {
                return Err(
                    "weights and beta must be non-empty square tensors with matching shapes",
                );
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
                    if !weight.is_finite() || weight < 0.0 {
                        return Err("mixture weights must be finite and non-negative");
                    }
                    let rate = lambda[i][j][k];
                    if !rate.is_finite() || rate <= 0.0 {
                        return Err("mixture rates must be finite and positive");
                    }
                    total_weight += weight;
                }
                if !total_weight.is_finite() || total_weight <= 0.0 {
                    return Err("mixture weights must sum to a finite positive value");
                }
                let last_positive = weights[i][j]
                    .iter()
                    .rposition(|&weight| weight > 0.0)
                    .unwrap();
                let mut cumulative = 0.0;
                for weight in &mut weights[i][j] {
                    cumulative += *weight / total_weight;
                    *weight = cumulative;
                }
                // Assign rounding residue to a positive component, never a trailing zero weight.
                weights[i][j][last_positive..].fill(1.0);
            }
        }
        Ok(Self {
            lambda,
            cdf: weights,
        })
    }

    fn sample_component<R: rand::Rng + ?Sized>(&self, i: usize, j: usize, rng: &mut R) -> usize {
        let cdf = &self.cdf[i][j];
        let u: f64 = rand::Rng::random(rng);
        cdf.iter()
            .position(|&threshold| u < threshold)
            .unwrap_or(cdf.len() - 1)
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

pub struct LaggedGammaKernel {
    gamma: GammaKernel,
    tau: Vec<Vec<f64>>,
}

impl LaggedGammaKernel {
    pub fn new(
        shape: Vec<Vec<f64>>,
        rate: Vec<Vec<f64>>,
        tau: Vec<Vec<f64>>,
    ) -> Result<Self, &'static str> {
        let d = shape.len();
        if d == 0 || rate.len() != d || tau.len() != d {
            return Err("shape, rate, and tau must be non-empty square matrices with matching shapes");
        }
        for i in 0..d {
            if shape[i].len() != d || rate[i].len() != d || tau[i].len() != d {
                return Err(
                    "shape, rate, and tau must be non-empty square matrices with matching shapes",
                );
            }
            for j in 0..d {
                if !shape[i][j].is_finite() || shape[i][j] <= 0.0 {
                    return Err("shape must contain only finite positive values");
                }
                if !rate[i][j].is_finite() || rate[i][j] <= 0.0 {
                    return Err("rate must contain only finite positive values");
                }
                if !tau[i][j].is_finite() || tau[i][j] < 0.0 {
                    return Err("tau must contain only finite non-negative values");
                }
            }
        }
        Ok(Self {
            gamma: GammaKernel::new(shape, rate),
            tau,
        })
    }
}

impl Kernel for LaggedGammaKernel {
    fn sample_delay<R: rand::Rng + ?Sized>(&self, i: usize, j: usize, rng: &mut R) -> f64 {
        self.tau[i][j] + self.gamma.sample_delay(i, j, rng)
    }

    fn validate_dimension(&self, dimension: usize) -> Result<(), &'static str> {
        if self.gamma.shape.len() != dimension {
            return Err("shape, rate, and tau dimensions must match the baseline dimension");
        }
        Ok(())
    }
}

impl Kernel for MixedExpKernel {
    fn sample_delay<R: rand::Rng + ?Sized>(&self, i: usize, j: usize, rng: &mut R) -> f64 {
        let idx = self.sample_component(i, j, rng);
        let lambda = self.lambda[i][j][idx];
        rand_distr::Exp::new(lambda).unwrap().sample(rng)
    }

    fn validate_dimension(&self, dimension: usize) -> Result<(), &'static str> {
        if self.lambda.len() != dimension {
            return Err("weights and beta dimensions must match the baseline dimension");
        }
        Ok(())
    }
}

pub struct LaggedMixedExpKernel {
    mixture: MixedExpKernel,
    tau: Vec<Vec<Vec<f64>>>,
}

impl LaggedMixedExpKernel {
    pub fn new(
        weights: Vec<Vec<Vec<f64>>>,
        lambda: Vec<Vec<Vec<f64>>>,
        tau: Vec<Vec<Vec<f64>>>,
    ) -> Result<Self, &'static str> {
        let mixture = MixedExpKernel::new(weights, lambda)?;
        let d = mixture.lambda.len();
        if tau.len() != d {
            return Err("tau must match the weights and beta shapes");
        }
        for i in 0..d {
            if tau[i].len() != d {
                return Err("tau must match the weights and beta shapes");
            }
            for j in 0..d {
                if tau[i][j].len() != mixture.lambda[i][j].len() {
                    return Err("tau must match the weights and beta shapes");
                }
                if tau[i][j].iter().any(|&lag| !lag.is_finite() || lag < 0.0) {
                    return Err("tau must contain only finite non-negative values");
                }
            }
        }
        Ok(Self { mixture, tau })
    }
}

impl Kernel for LaggedMixedExpKernel {
    fn sample_delay<R: rand::Rng + ?Sized>(&self, i: usize, j: usize, rng: &mut R) -> f64 {
        let idx = self.mixture.sample_component(i, j, rng);
        let exponential_delay = rand_distr::Exp::new(self.mixture.lambda[i][j][idx])
            .unwrap()
            .sample(rng);
        self.tau[i][j][idx] + exponential_delay
    }

    fn validate_dimension(&self, dimension: usize) -> Result<(), &'static str> {
        self.mixture.validate_dimension(dimension)
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
    LaggedGamma(LaggedGammaKernel),
    MixedExponential(MixedExpKernel),
    LaggedMixedExponential(LaggedMixedExpKernel),
    PowerLaw(PowerLawKernel),
}

impl Kernel for KernelKind {
    fn sample_delay<R: rand::Rng + ?Sized>(&self, i: usize, j: usize, rng: &mut R) -> f64 {
        match self {
            Self::Exponential(kernel) => kernel.sample_delay(i, j, rng),
            Self::LaggedExponential(kernel) => kernel.sample_delay(i, j, rng),
            Self::Gamma(kernel) => kernel.sample_delay(i, j, rng),
            Self::LaggedGamma(kernel) => kernel.sample_delay(i, j, rng),
            Self::MixedExponential(kernel) => kernel.sample_delay(i, j, rng),
            Self::LaggedMixedExponential(kernel) => kernel.sample_delay(i, j, rng),
            Self::PowerLaw(kernel) => kernel.sample_delay(i, j, rng),
        }
    }

    fn validate_dimension(&self, dimension: usize) -> Result<(), &'static str> {
        match self {
            Self::Exponential(kernel) => kernel.validate_dimension(dimension),
            Self::LaggedExponential(kernel) => kernel.validate_dimension(dimension),
            Self::Gamma(kernel) => kernel.validate_dimension(dimension),
            Self::LaggedGamma(kernel) => kernel.validate_dimension(dimension),
            Self::MixedExponential(kernel) => kernel.validate_dimension(dimension),
            Self::LaggedMixedExponential(kernel) => kernel.validate_dimension(dimension),
            Self::PowerLaw(kernel) => kernel.validate_dimension(dimension),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        GammaKernel, Kernel, KernelKind, LaggedExpKernel, LaggedGammaKernel, LaggedMixedExpKernel,
        MixedExpKernel,
    };
    use rand::{RngCore, SeedableRng};
    use rand_distr::Distribution;

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

    #[test]
    fn lagged_gamma_delay_has_expected_support_mean_and_variance() {
        for shape in [0.5, 1.0, 2.5] {
            let rate = 2.0;
            let lag = 0.4;
            let kernel =
                LaggedGammaKernel::new(vec![vec![shape]], vec![vec![rate]], vec![vec![lag]])
                    .unwrap();
            let mut rng = rand::rngs::StdRng::seed_from_u64(20261005);
            let sample_count = 50_000;
            let mut total = 0.0;
            let mut total_squared = 0.0;

            for _ in 0..sample_count {
                let delay = kernel.sample_delay(0, 0, &mut rng);
                assert!(delay >= lag);
                total += delay;
                total_squared += delay * delay;
            }

            let sample_mean = total / sample_count as f64;
            let sample_variance = total_squared / sample_count as f64 - sample_mean * sample_mean;
            assert!((sample_mean - (lag + shape / rate)).abs() < 0.02);
            assert!((sample_variance - shape / (rate * rate)).abs() < 0.03);
        }
    }

    #[test]
    fn lagged_gamma_delay_has_expected_cdf() {
        let kernel =
            LaggedGammaKernel::new(vec![vec![2.0]], vec![vec![2.0]], vec![vec![0.4]])
                .unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(20261006);
        let sample_count = 50_000;
        let thresholds: [f64; 3] = [0.4, 0.9, 1.9];
        let mut below = [0; 3];
        for _ in 0..sample_count {
            let delay = kernel.sample_delay(0, 0, &mut rng);
            for (idx, &threshold) in thresholds.iter().enumerate() {
                if delay <= threshold {
                    below[idx] += 1;
                }
            }
        }
        for (idx, &threshold) in thresholds.iter().enumerate() {
            let x = 2.0 * (threshold - 0.4);
            let expected_cdf = 1.0 - (-x).exp() * (1.0 + x);
            assert!((below[idx] as f64 / sample_count as f64 - expected_cdf).abs() < 0.01);
        }
    }

    #[test]
    fn lagged_gamma_uses_parent_child_indices() {
        let kernel = LaggedGammaKernel::new(
            vec![vec![1.0, 0.5], vec![2.5, 3.0]],
            vec![vec![2.0, 3.0], vec![5.0, 7.0]],
            vec![vec![0.2, 10.0], vec![20.0, 30.0]],
        )
        .unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(47);
        let mut reference_rng = rng.clone();
        for (parent, child, shape, rate, lag) in [(0, 1, 0.5, 3.0, 10.0), (1, 0, 2.5, 5.0, 20.0)] {
            for _ in 0..100 {
                let expected = lag
                    + rand_distr::Gamma::new(shape, 1.0 / rate)
                        .unwrap()
                        .sample(&mut reference_rng);
                assert_eq!(kernel.sample_delay(parent, child, &mut rng), expected);
            }
        }
    }

    #[test]
    fn zero_lag_gamma_matches_gamma_random_stream() {
        for shape in [0.5, 1.0, 2.5] {
            let gamma = GammaKernel::new(vec![vec![shape]], vec![vec![2.0]]);
            let lagged =
                LaggedGammaKernel::new(vec![vec![shape]], vec![vec![2.0]], vec![vec![0.0]])
                    .unwrap();
            let mut rng_gamma = rand::rngs::StdRng::seed_from_u64(89);
            let mut rng_lagged = rng_gamma.clone();
            for _ in 0..500 {
                assert_eq!(
                    gamma.sample_delay(0, 0, &mut rng_gamma).to_bits(),
                    lagged.sample_delay(0, 0, &mut rng_lagged).to_bits()
                );
            }
            assert_eq!(rng_gamma.next_u64(), rng_lagged.next_u64());
        }
    }

    #[test]
    fn lagged_gamma_rejects_invalid_shapes_values_and_dimensions() {
        let valid = vec![vec![1.0]];
        for invalid in [vec![], vec![vec![]], vec![vec![1.0, 2.0]], vec![vec![1.0]; 2]] {
            assert!(LaggedGammaKernel::new(invalid.clone(), valid.clone(), valid.clone()).is_err());
            assert!(LaggedGammaKernel::new(valid.clone(), invalid.clone(), valid.clone()).is_err());
            assert!(LaggedGammaKernel::new(valid.clone(), valid.clone(), invalid).is_err());
        }
        assert!(LaggedGammaKernel::new(vec![], vec![], vec![]).is_err());
        for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                LaggedGammaKernel::new(vec![vec![invalid]], valid.clone(), valid.clone()).is_err()
            );
            assert!(
                LaggedGammaKernel::new(valid.clone(), vec![vec![invalid]], valid.clone()).is_err()
            );
        }
        for invalid in [-0.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                LaggedGammaKernel::new(valid.clone(), valid.clone(), vec![vec![invalid]]).is_err()
            );
        }
        let kernel = KernelKind::LaggedGamma(
            LaggedGammaKernel::new(valid.clone(), valid, vec![vec![0.0]]).unwrap(),
        );
        assert!(kernel.validate_dimension(1).is_ok());
        assert!(kernel.validate_dimension(0).is_err());
        assert!(kernel.validate_dimension(2).is_err());
    }

    #[test]
    fn lagged_mixture_has_expected_support_cdf_and_mean() {
        let kernel = LaggedMixedExpKernel::new(
            vec![vec![vec![2.0, 3.0]]],
            vec![vec![vec![2.0, 0.5]]],
            vec![vec![vec![0.2, 1.4]]],
        )
        .unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(20260905);
        let sample_count = 50_000;
        let thresholds = [0.8, 2.0];
        let mut below = [0; 2];
        let mut total = 0.0;

        for _ in 0..sample_count {
            let delay = kernel.sample_delay(0, 0, &mut rng);
            assert!(delay > 0.2);
            total += delay;
            for (idx, &threshold) in thresholds.iter().enumerate() {
                if delay <= threshold {
                    below[idx] += 1;
                }
            }
        }

        let expected_mean = 0.4 * (0.2 + 1.0 / 2.0) + 0.6 * (1.4 + 1.0 / 0.5);
        assert!((total / sample_count as f64 - expected_mean).abs() < 0.06);
        for (idx, &threshold) in thresholds.iter().enumerate() {
            let expected_cdf = 0.4 * (1.0 - (-2.0 * (threshold - 0.2)).exp())
                + if threshold > 1.4 {
                    0.6 * (1.0 - (-0.5 * (threshold - 1.4)).exp())
                } else {
                    0.0
                };
            assert!((below[idx] as f64 / sample_count as f64 - expected_cdf).abs() < 0.01);
        }
    }

    #[test]
    fn lagged_mixture_uses_parent_child_and_component_indices() {
        let kernel = LaggedMixedExpKernel::new(
            vec![
                vec![vec![1.0], vec![0.0, 1.0]],
                vec![vec![1.0, 0.0, 0.0], vec![1.0]],
            ],
            vec![
                vec![vec![1.0], vec![3.0, 5.0]],
                vec![vec![7.0, 8.0, 9.0], vec![11.0]],
            ],
            vec![
                vec![vec![0.2], vec![10.0, 20.0]],
                vec![vec![30.0, 40.0, 50.0], vec![60.0]],
            ],
        )
        .unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(47);
        let mut reference_rng = rng.clone();
        for (parent, child, rate, lag) in [(0, 1, 5.0, 20.0), (1, 0, 7.0, 30.0)] {
            for _ in 0..100 {
                let _: f64 = rand::Rng::random(&mut reference_rng);
                let expected = lag
                    + rand_distr::Exp::new(rate)
                        .unwrap()
                        .sample(&mut reference_rng);
                assert_eq!(kernel.sample_delay(parent, child, &mut rng), expected);
            }
        }
    }

    #[test]
    fn zero_lag_mixture_matches_mixed_exponential_random_stream() {
        for weights in [vec![1.0], vec![0.0, 2.0, 3.0], vec![0.1, 0.2, 0.3]] {
            let beta: Vec<f64> = (1..=weights.len()).map(|k| k as f64).collect();
            let tau = vec![0.0; weights.len()];
            let mixed =
                MixedExpKernel::new(vec![vec![weights.clone()]], vec![vec![beta.clone()]]).unwrap();
            let lagged =
                LaggedMixedExpKernel::new(vec![vec![weights]], vec![vec![beta]], vec![vec![tau]])
                    .unwrap();
            let mut rng_mixed = rand::rngs::StdRng::seed_from_u64(89);
            let mut rng_lagged = rng_mixed.clone();
            for _ in 0..500 {
                assert_eq!(
                    mixed.sample_delay(0, 0, &mut rng_mixed).to_bits(),
                    lagged.sample_delay(0, 0, &mut rng_lagged).to_bits()
                );
            }
            assert_eq!(rng_mixed.next_u64(), rng_lagged.next_u64());
        }
    }

    #[test]
    #[allow(deprecated)] // rand's deterministic testing RNG has no replacement.
    fn mixture_selection_skips_zero_weights_at_boundaries() {
        let mixed = MixedExpKernel::new(
            vec![vec![vec![0.0, 1.0, 0.0, 1.0, 0.0]]],
            vec![vec![vec![1.0; 5]]],
        )
        .unwrap();
        for (bits, expected) in [(0, 1), (1 << 63, 3), (u64::MAX, 3)] {
            let mut rng = rand::rngs::mock::StepRng::new(bits, 0);
            assert_eq!(mixed.sample_component(0, 0, &mut rng), expected);
        }
        let mut weights = vec![1.0; 10];
        weights.push(0.0);
        let mixed = MixedExpKernel::new(vec![vec![weights]], vec![vec![vec![1.0; 11]]]).unwrap();
        let mut rng = rand::rngs::mock::StepRng::new(u64::MAX, 0);
        assert_eq!(mixed.sample_component(0, 0, &mut rng), 9);
    }

    #[test]
    fn mixtures_reject_invalid_shapes_values_and_dimensions() {
        let tensor = |values: Vec<f64>| vec![vec![values]];
        for weights in [vec![], vec![vec![]], tensor(vec![])] {
            assert!(MixedExpKernel::new(weights.clone(), weights).is_err());
        }
        assert!(MixedExpKernel::new(tensor(vec![1.0, 2.0]), tensor(vec![1.0])).is_err());
        for weights in [
            vec![-1.0],
            vec![f64::NAN],
            vec![f64::INFINITY],
            vec![0.0],
            vec![f64::MAX, f64::MAX],
        ] {
            let beta = vec![1.0; weights.len()];
            assert!(MixedExpKernel::new(tensor(weights), tensor(beta)).is_err());
        }
        for beta in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(MixedExpKernel::new(tensor(vec![1.0]), tensor(vec![beta])).is_err());
        }
        for tau in [
            vec![],
            vec![vec![]],
            tensor(vec![]),
            tensor(vec![0.0, 0.1]),
            tensor(vec![-0.1]),
            tensor(vec![f64::NAN]),
            tensor(vec![f64::INFINITY]),
        ] {
            assert!(LaggedMixedExpKernel::new(tensor(vec![1.0]), tensor(vec![1.0]), tau).is_err());
        }
        let mixed = MixedExpKernel::new(tensor(vec![1.0]), tensor(vec![1.0])).unwrap();
        let lagged =
            LaggedMixedExpKernel::new(tensor(vec![1.0]), tensor(vec![1.0]), tensor(vec![0.0]))
                .unwrap();
        assert!(mixed.validate_dimension(1).is_ok());
        assert!(lagged.validate_dimension(1).is_ok());
        assert!(mixed.validate_dimension(2).is_err());
        assert!(lagged.validate_dimension(2).is_err());
    }
}
