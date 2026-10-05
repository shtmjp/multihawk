from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.special import gammainc

from multihawk import BaselineSpec, KernelSpec, simulate_hawkes

if TYPE_CHECKING:
    from multihawk.simulation import SamplingMethod


def _kernel_params() -> dict[str, list[list[float]]]:
    return {
        "shape": [[0.4, 2.5], [1.0, 3.0]],
        "rate": [[1.5, 2.0], [0.8, 4.0]],
        "tau": [[0.1, 0.5], [0.3, 0.0]],
    }


@pytest.mark.parametrize("bitgen", [np.random.PCG64, np.random.PCG64DXSM])
@pytest.mark.parametrize("sampling_method", ["thinning", "inverse_transform"])
@pytest.mark.parametrize(
    "baseline",
    [
        BaselineSpec(kind="constant", params={"values": [0.7, 0.5]}),
        BaselineSpec(
            kind="piecewise_linear",
            params={
                "breaks": [0.0, 50.0, 100.0],
                "values": [[0.7, 0.9, 0.7], [0.5, 0.3, 0.5]],
            },
        ),
    ],
)
def test_zero_lag_matches_gamma_output_and_rng(
    bitgen: type[np.random.BitGenerator], sampling_method: SamplingMethod, baseline: BaselineSpec
) -> None:
    params = _kernel_params()
    params["tau"] = [[0.0, 0.0], [0.0, 0.0]]
    gamma = KernelSpec(kind="gamma", params={"shape": params["shape"], "rate": params["rate"]})
    lagged = KernelSpec(kind="lagged_gamma", params=params)
    rng_gamma = np.random.Generator(bitgen(101))
    rng_lagged = np.random.Generator(bitgen(101))
    alpha = [[0.3, 0.2], [0.1, 0.25]]
    expected = simulate_hawkes(
        t_max=100.0,
        baseline=baseline,
        alpha=alpha,
        kernel=gamma,
        rng=rng_gamma,
        sampling_method=sampling_method,
    )
    actual = simulate_hawkes(
        t_max=100.0,
        baseline=baseline,
        alpha=alpha,
        kernel=lagged,
        rng=rng_lagged,
        sampling_method=sampling_method,
    )
    assert actual == expected
    assert rng_lagged.bit_generator.state == rng_gamma.bit_generator.state


@pytest.mark.parametrize("sampling_method", ["thinning", "inverse_transform"])
def test_integer_seed_is_reproducible(sampling_method: SamplingMethod) -> None:
    baseline = BaselineSpec(kind="constant", params={"values": [0.7, 0.5]})
    kernel = KernelSpec(kind="lagged_gamma", params=_kernel_params())
    first = simulate_hawkes(
        t_max=100.0,
        baseline=baseline,
        alpha=[[0.3, 0.2], [0.1, 0.25]],
        kernel=kernel,
        seed=20261005,
        sampling_method=sampling_method,
    )
    second = simulate_hawkes(
        t_max=100.0,
        baseline=baseline,
        alpha=[[0.3, 0.2], [0.1, 0.25]],
        kernel=kernel,
        seed=20261005,
        sampling_method=sampling_method,
    )
    assert first == second
    assert first.obs_window == (0.0, 100.0)
    assert all(ts == sorted(ts) for ts in first.timestamps)
    assert all(0.0 <= t < first.obs_window[1] for ts in first.timestamps for t in ts)


@pytest.mark.parametrize("bitgen", [np.random.PCG64, np.random.PCG64DXSM])
def test_rng_advances_and_restores(bitgen: type[np.random.BitGenerator]) -> None:
    baseline = BaselineSpec(kind="constant", params={"values": [0.7, 0.5]})
    kernel = KernelSpec(kind="lagged_gamma", params=_kernel_params())
    rng = np.random.Generator(bitgen(42))
    before = rng.bit_generator.state
    expected = simulate_hawkes(
        t_max=100.0,
        baseline=baseline,
        alpha=[[0.3, 0.2], [0.1, 0.25]],
        kernel=kernel,
        rng=rng,
    )
    after = rng.bit_generator.state
    assert after != before
    rng.bit_generator.state = before
    restored = simulate_hawkes(
        t_max=100.0,
        baseline=baseline,
        alpha=[[0.3, 0.2], [0.1, 0.25]],
        kernel=kernel,
        rng=rng,
    )
    assert restored == expected
    assert rng.bit_generator.state == after


@pytest.mark.parametrize("sampling_method", ["thinning", "inverse_transform"])
@pytest.mark.parametrize("lag", [0.5, 3.0, 4.0])
def test_parent_child_orientation_and_horizon_cutoff(
    sampling_method: SamplingMethod, lag: float
) -> None:
    # Only type 0 has immigrants, and only the [0][1] pair produces offspring.
    immigrant_stop = 0.1
    baseline = BaselineSpec(
        kind="piecewise_constant",
        params={"breaks": [0.0, immigrant_stop, 3.0], "rates": [[1000.0, 0.0], [0.0, 0.0]]},
    )
    kernel = KernelSpec(
        kind="lagged_gamma",
        params={
            "shape": [[2.0, 0.4], [3.0, 1.0]],
            "rate": [[1.0, 100.0], [0.5, 1.0]],
            "tau": [[4.0, lag], [4.0, 4.0]],
        },
    )
    result = simulate_hawkes(
        t_max=3.0,
        baseline=baseline,
        alpha=[[0.0, 2.0], [0.0, 0.0]],
        kernel=kernel,
        seed=123,
        sampling_method=sampling_method,
    )
    parents, children = result.timestamps
    assert parents
    assert all(0.0 <= t < immigrant_stop for t in parents)
    if lag >= result.obs_window[1]:
        assert children == []
    else:
        assert children
        assert all(parents[0] + lag <= t < result.obs_window[1] for t in children)


@pytest.mark.parametrize("shape", [0.4, 2.5])
@pytest.mark.parametrize("sampling_method", ["thinning", "inverse_transform"])
def test_finite_horizon_child_count_matches_lagged_gamma_mass(
    shape: float, sampling_method: SamplingMethod
) -> None:
    # Conditional on the immigrants, retained children have Poisson mean
    # alpha * sum_parent GammaCDF(t_max - parent_time - tau; shape, rate).
    # Include parents on both sides of t_max - tau to exercise zero remaining mass.
    t_max, lag, rate, excitation = 3.0, 1.5, 0.7, 3.0
    baseline = BaselineSpec(kind="constant", params={"values": [1000.0, 0.0]})
    kernel = KernelSpec(
        kind="lagged_gamma",
        params={
            "shape": [[1.0, shape], [7.0, 1.0]],
            "rate": [[1.0, rate], [5.0, 1.0]],
            "tau": [[0.0, lag], [0.0, 0.0]],
        },
    )
    result = simulate_hawkes(
        t_max=t_max,
        baseline=baseline,
        alpha=[[0.0, excitation], [0.0, 0.0]],
        kernel=kernel,
        seed=20261005,
        sampling_method=sampling_method,
    )
    parents, children = result.timestamps
    remaining = np.maximum(t_max - np.asarray(parents) - lag, 0.0)
    mean = float(excitation * np.sum(gammainc(shape, rate * remaining)))
    assert mean > 0.0
    assert abs(len(children) - mean) < 6.0 * np.sqrt(mean)
    assert all(lag <= t < t_max for t in children)


@pytest.mark.parametrize("missing", ["shape", "rate", "tau"])
def test_required_parameters(missing: str) -> None:
    params = _kernel_params()
    del params[missing]
    with pytest.raises(ValueError, match="requires"):
        KernelSpec(kind="lagged_gamma", params=params).to_backend()


@pytest.mark.parametrize("name", ["shape", "rate"])
@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_rejects_invalid_shape_and_rate(name: str, value: float) -> None:
    params = {"shape": [[1.0]], "rate": [[1.0]], "tau": [[0.0]]}
    params[name] = [[value]]
    with pytest.raises(ValueError, match=name):
        simulate_hawkes(
            t_max=1.0,
            baseline=BaselineSpec(kind="constant", params={"values": [0.0]}),
            alpha=[[0.0]],
            kernel=KernelSpec(kind="lagged_gamma", params=params),
            seed=1,
        )


@pytest.mark.parametrize("tau", [-1.0, np.nan, np.inf, -np.inf])
def test_rejects_invalid_lags(tau: float) -> None:
    with pytest.raises(ValueError, match="tau"):
        simulate_hawkes(
            t_max=1.0,
            baseline=BaselineSpec(kind="constant", params={"values": [0.0]}),
            alpha=[[0.0]],
            kernel=KernelSpec(
                kind="lagged_gamma", params={"shape": [[1.0]], "rate": [[1.0]], "tau": [[tau]]}
            ),
            seed=1,
        )


@pytest.mark.parametrize("name", ["shape", "rate", "tau"])
@pytest.mark.parametrize(
    "matrix", [[], [[]], [[1.0, 2.0]], [[1.0], [2.0]], [[1.0, 2.0], [3.0, 4.0]]]
)
def test_rejects_mismatched_or_empty_shapes(name: str, matrix: list[list[float]]) -> None:
    params = {"shape": [[1.0]], "rate": [[1.0]], "tau": [[0.0]]}
    params[name] = matrix
    with pytest.raises(ValueError, match=r"shape|square|empty"):
        simulate_hawkes(
            t_max=1.0,
            baseline=BaselineSpec(kind="constant", params={"values": [0.0]}),
            alpha=[[0.0]],
            kernel=KernelSpec(kind="lagged_gamma", params=params),
            seed=1,
        )


def test_kernel_dimension_must_match_baseline() -> None:
    kernel = KernelSpec(
        kind="lagged_gamma", params={"shape": [[1.0]], "rate": [[1.0]], "tau": [[0.0]]}
    )
    with pytest.raises(ValueError, match="dimension"):
        simulate_hawkes(
            t_max=1.0,
            baseline=BaselineSpec(kind="constant", params={"values": [0.0, 0.0]}),
            alpha=[[0.0, 0.0], [0.0, 0.0]],
            kernel=kernel,
            seed=1,
        )


def test_numpy_parameters_match_nested_sequences() -> None:
    params = _kernel_params()
    arrays = {name: np.asarray(values) for name, values in params.items()}
    nested = KernelSpec(kind="lagged_gamma", params=params)
    numpy_kernel = KernelSpec(kind="lagged_gamma", params=arrays)
    assert numpy_kernel.to_backend() == nested.to_backend()
