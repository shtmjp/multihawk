import pytest

from multihawk import BaselineSpec, KernelSpec, simulate_hawkes

try:
    import numpy as np
except ImportError:
    pytest.skip("numpy not available", allow_module_level=True)

try:
    import Hawkes as hk  # noqa: N813
except ImportError:
    pytest.skip("hawkes not available", allow_module_level=True)


@pytest.mark.external
def test_univariate_mixture_exp_plinear_baseline_hawkes() -> None:
    # True values from examples/mle.ipynb
    mu_vals = [0.1, 0.3, 0.1]  # baseline values at breakpoints (0, T/2, T)
    alpha_total = 0.2
    weights = [0.6, 0.4]
    lambdas = [0.5, 2.0]

    T = 100_000.0
    rng = np.random.default_rng(0)

    # multihawk setup: 1D, piecewise-linear baseline, 2-exp mixture kernel
    baseline = BaselineSpec(
        kind="piecewise_linear",
        params={
            "breaks": [0.0, T / 2.0, T],
            "values": [mu_vals],
        },
    )
    kernel = KernelSpec(
        kind="mixed_exponential",
        params={
            "weights": [[weights]],  # shape [M, M, R] with M=1
            "beta": [[lambdas]],
        },
    )
    alpha = [[alpha_total]]

    result = simulate_hawkes(t_max=T, baseline=baseline, alpha=alpha, kernel=kernel, rng=rng)
    events = result.timestamps[0]

    # Fit Hawkes model: mixture of exponentials with piecewise-linear baseline
    model = (
        hk.estimator()
        .set_kernel("exp", num_exp=len(lambdas))
        .set_baseline("plinear", num_basis=len(mu_vals))
    )
    model.fit(events, [0.0, T])

    para = getattr(model, "para", None)
    if para is None:  # pragma: no cover - version difference
        para = getattr(model, "get_para", lambda: None)()
    if para is None:  # pragma: no cover - unexpected API change
        pytest.skip("hawkes model has no parameters available")

    # Expected values mapping
    alpha_parts = [alpha_total * w for w in weights]

    rtol_mu = 0.35
    atol_mu = 0.05
    rtol_alpha = 0.5
    atol_alpha = 0.05
    rtol_beta = 0.2
    atol_beta = 0.05

    mu_hat = np.asarray(para.get("mu"), dtype=float)
    alpha_hat = np.asarray(para.get("alpha"), dtype=float)
    beta_hat = np.asarray(para.get("beta"), dtype=float)

    assert np.allclose(mu_hat, np.asarray(mu_vals, dtype=float), rtol=rtol_mu, atol=atol_mu)
    assert np.allclose(
        alpha_hat, np.asarray(alpha_parts, dtype=float), rtol=rtol_alpha, atol=atol_alpha
    )
    assert np.allclose(beta_hat, np.asarray(lambdas, dtype=float), rtol=rtol_beta, atol=atol_beta)
