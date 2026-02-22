import pytest

from multihawk import BaselineSpec, KernelSpec, simulate_hawkes

try:
    import numpy as np
except ImportError:
    pytest.skip("numpy not available", allow_module_level=True)

try:
    import hawkesbook
except ImportError:
    pytest.skip("hawkesbook not available", allow_module_level=True)


@pytest.mark.external
def test_univariate_exponential_mle_em_hawkesbook() -> None:
    # Ground-truth parameters (multihawk parameterization)
    mu = 0.2
    alpha = 0.4
    beta = 2.0

    # hawkesbook parameterization for exponential kernel
    # alpha_in_book is the amplitude for exp(-beta t)
    lambda_in_book = mu
    alpha_in_book = alpha * beta
    beta_in_book = beta

    rng = np.random.default_rng(0)
    baseline = BaselineSpec(kind="constant", params={"values": [mu]})
    kernel = KernelSpec(kind="exponential", params={"beta": [[beta]]})
    T = 10_000.0

    data = simulate_hawkes(t_max=T, baseline=baseline, alpha=[[alpha]], kernel=kernel, rng=rng)
    t = np.asarray(data.timestamps[0], dtype=float)

    # hawkesbook APIs
    mle = hawkesbook.exp_mle(t, T)
    em = hawkesbook.exp_em(t, T)

    mle = np.asarray(mle, dtype=float)
    em = np.asarray(em, dtype=float)

    # Tolerances: allow moderate relative error, small absolute wiggle
    rtol = 0.25
    atol = 0.02

    assert np.allclose(mle[0], lambda_in_book, rtol=rtol, atol=atol)
    assert np.allclose(mle[1], alpha_in_book, rtol=rtol, atol=atol)
    assert np.allclose(mle[2], beta_in_book, rtol=rtol, atol=atol)

    assert np.allclose(em[0], lambda_in_book, rtol=rtol, atol=atol)
    assert np.allclose(em[1], alpha_in_book, rtol=rtol, atol=atol)
    assert np.allclose(em[2], beta_in_book, rtol=rtol, atol=atol)
