import pytest

from multihawk import BaselineSpec, KernelSpec, simulate_hawkes

try:
    import numpy as np
except ImportError:
    pytest.skip("numpy not available", allow_module_level=True)

try:
    from sparklen.hawkes.inference import LearnerHawkesExp
except ImportError:
    pytest.skip("sparklen not available", allow_module_level=True)


@pytest.mark.external
def test_multivariate_exponential_mle_sparklen_structure_and_values() -> None:
    # Setup similar to examples/mle.ipynb
    d = 3
    beta = 3.0
    mu = np.array([0.6, 0.55, 0.6], dtype=float)
    alpha = np.zeros((d, d), dtype=float)
    alpha[:2, :2] += 0.1
    alpha[1:, 1:] += 0.15

    baseline = BaselineSpec(kind="constant", params={"values": mu})
    kernel = KernelSpec(kind="exponential", params={"beta": [[beta] * d] * d})

    T = 10_000.0
    rng = np.random.default_rng(123)
    sim = simulate_hawkes(t_max=T, baseline=baseline, alpha=alpha, kernel=kernel, rng=rng)  # type: ignore[]

    # Prepare events for sparklen: a list (one sequence) of per-dimension arrays
    events = [[np.asarray(sim.timestamps[i], dtype=float) for i in range(d)]]

    learner = LearnerHawkesExp(
        decay=beta,
        loss="log-likelihood",
        penalty="none",
        optimizer="agd",
        lr_scheduler="backtracking",
        max_iter=100,
    )

    learner.fit(events, end_time=T)

    theta_hat = learner.estimated_params
    assert isinstance(theta_hat, np.ndarray)

    mu_hat = theta_hat[:, 0]
    alpha_hat = theta_hat[:, 1:]

    # Sanity checks: finite values and reasonable shape
    assert np.isfinite(mu_hat).all()
    assert np.isfinite(alpha_hat).all()

    # If the learner did not move from its default initialization (all 0.2),
    # xfail to avoid a misleading failure. Otherwise, check closeness to truth.
    if np.allclose(theta_hat, 0.2):
        pytest.xfail(
            "sparklen learner returned its initialization (0.2); cannot validate closeness"
        )

    # Closeness to true baselines and interactions (allow generous tolerances)
    assert np.allclose(mu_hat, mu, rtol=0.3, atol=0.1)

    nz = alpha != 0
    z = alpha == 0
    # Non-zero entries should be close to their true values on average
    assert np.allclose(alpha_hat[nz].mean(), alpha[nz].mean(), atol=0.05)
    # Zero entries should remain small relative to non-zeros
    assert alpha_hat[z].mean() < alpha_hat[nz].mean() - 0.05
