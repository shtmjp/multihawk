from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from multihawk import BaselineSpec, KernelSpec, simulate_hawkes

if TYPE_CHECKING:
    from typing import Literal

    from multihawk.simulation import SamplingMethod


def _kernel_params() -> dict[str, list[list[list[float]]]]:
    # Different component counts exercise the nested, per-pair representation.
    return {
        "weights": [[[2.0, 0.0, 3.0], [1.0]], [[4.0, 1.0], [0.2, 0.5, 0.3]]],
        "beta": [[[1.0, 2.0, 3.0], [2.0]], [[0.5, 1.5], [1.0, 2.0, 4.0]]],
        "tau": [[[0.1, 0.2, 0.3], [0.5]], [[0.3, 0.7], [0.0, 0.2, 0.9]]],
    }


@pytest.mark.parametrize("bitgen", [np.random.PCG64, np.random.PCG64DXSM])
@pytest.mark.parametrize("sampling_method", ["thinning", "inverse_transform"])
def test_zero_lag_matches_mixed_exponential_output_and_rng(
    bitgen: type[np.random.BitGenerator], sampling_method: SamplingMethod
) -> None:
    params = _kernel_params()
    params["tau"] = [[[0.0] * len(pair) for pair in row] for row in params["weights"]]
    mixed = KernelSpec(
        kind="mixed_exponential",
        params={"weights": params["weights"], "beta": params["beta"]},
    )
    lagged = KernelSpec(kind="lagged_mixed_exponential", params=params)
    rng_mixed = np.random.Generator(bitgen(101))
    rng_lagged = np.random.Generator(bitgen(101))
    baseline = BaselineSpec(kind="constant", params={"values": [0.7, 0.5]})
    alpha = [[0.3, 0.2], [0.1, 0.25]]
    expected = simulate_hawkes(
        t_max=100.0,
        baseline=baseline,
        alpha=alpha,
        kernel=mixed,
        rng=rng_mixed,
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
    assert rng_lagged.bit_generator.state == rng_mixed.bit_generator.state


@pytest.mark.parametrize("sampling_method", ["thinning", "inverse_transform"])
def test_integer_seed_is_reproducible(sampling_method: SamplingMethod) -> None:
    baseline = BaselineSpec(kind="constant", params={"values": [0.7, 0.5]})
    kernel = KernelSpec(kind="lagged_mixed_exponential", params=_kernel_params())
    first = simulate_hawkes(
        t_max=100.0,
        baseline=baseline,
        alpha=[[0.3, 0.2], [0.1, 0.25]],
        kernel=kernel,
        seed=20260905,
        sampling_method=sampling_method,
    )
    second = simulate_hawkes(
        t_max=100.0,
        baseline=baseline,
        alpha=[[0.3, 0.2], [0.1, 0.25]],
        kernel=kernel,
        seed=20260905,
        sampling_method=sampling_method,
    )
    assert first == second
    assert first.obs_window == (0.0, 100.0)
    assert all(ts == sorted(ts) for ts in first.timestamps)
    assert all(0.0 <= t < first.obs_window[1] for ts in first.timestamps for t in ts)


@pytest.mark.parametrize("bitgen", [np.random.PCG64, np.random.PCG64DXSM])
def test_rng_advances_and_restores(bitgen: type[np.random.BitGenerator]) -> None:
    baseline = BaselineSpec(kind="constant", params={"values": [0.7, 0.5]})
    kernel = KernelSpec(kind="lagged_mixed_exponential", params=_kernel_params())
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
    # Only type 0 has immigrants; it excites type 1 for one generation.
    immigrant_stop = 0.1
    baseline = BaselineSpec(
        kind="piecewise_constant",
        params={"breaks": [0.0, immigrant_stop, 3.0], "rates": [[1000.0, 0.0], [0.0, 0.0]]},
    )
    kernel = KernelSpec(
        kind="lagged_mixed_exponential",
        params={
            "weights": [[[1.0], [0.0, 2.0]], [[1.0], [1.0]]],
            "beta": [[[1.0], [1.0, 100.0]], [[1.0], [1.0]]],
            "tau": [[[4.0], [0.0, lag]], [[4.0], [4.0]]],
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


def test_finite_horizon_child_count_matches_lagged_mixture_mass() -> None:
    # Conditional on the immigrants, retained children are Poisson with mean
    # alpha * sum_parent CDF(t_max - parent_time). The late component is censored.
    baseline = BaselineSpec(
        kind="piecewise_constant",
        params={"breaks": [0.0, 0.1, 3.0], "rates": [[10000.0, 0.0], [0.0, 0.0]]},
    )
    kernel = KernelSpec(
        kind="lagged_mixed_exponential",
        params={
            "weights": [[[1.0], [1.0, 3.0]], [[1.0], [1.0]]],
            "beta": [[[1.0], [2.0, 4.0]], [[1.0], [1.0]]],
            "tau": [[[0.0], [0.5, 5.0]], [[0.0], [0.0]]],
        },
    )
    result = simulate_hawkes(
        t_max=3.0,
        baseline=baseline,
        alpha=[[0.0, 2.0], [0.0, 0.0]],
        kernel=kernel,
        seed=20260905,
    )
    parents, children = result.timestamps
    remaining = 3.0 - np.asarray(parents) - 0.5
    mean = float(2.0 * 0.25 * np.sum(-np.expm1(-2.0 * remaining)))
    assert mean > 0.0
    assert abs(len(children) - mean) < 6.0 * np.sqrt(mean)


@pytest.mark.parametrize("missing", ["weights", "beta", "tau"])
def test_required_parameters(missing: str) -> None:
    params = _kernel_params()
    del params[missing]
    with pytest.raises(ValueError, match="requires"):
        KernelSpec(kind="lagged_mixed_exponential", params=params).to_backend()


@pytest.mark.parametrize("kind", ["mixed_exponential", "lagged_mixed_exponential"])
@pytest.mark.parametrize(
    ("name", "values"),
    [
        ("weights", [-1.0, 2.0]),
        ("weights", [0.0, 0.0]),
        ("weights", [np.nan, 1.0]),
        ("weights", [np.inf, 1.0]),
        ("weights", [1e308, 1e308]),
        ("beta", [0.0, 1.0]),
        ("beta", [-1.0, 1.0]),
        ("beta", [np.inf, 1.0]),
        ("beta", [np.nan, 1.0]),
    ],
)
def test_mixed_kernels_reject_invalid_weights_and_rates(
    kind: Literal["mixed_exponential", "lagged_mixed_exponential"],
    name: str,
    values: list[float],
) -> None:
    params = {"weights": [[[1.0, 2.0]]], "beta": [[[1.0, 2.0]]], "tau": [[[0.0, 0.5]]]}
    params[name] = [[values]]
    with pytest.raises(ValueError, match=r"weights|beta|rates"):
        simulate_hawkes(
            t_max=1.0,
            baseline=BaselineSpec(kind="constant", params={"values": [1.0]}),
            alpha=[[0.0]],
            kernel=KernelSpec(kind=kind, params=params),
            seed=1,
        )


@pytest.mark.parametrize("tau", [-1.0, np.nan, np.inf, -np.inf])
def test_rejects_invalid_lags(tau: float) -> None:
    kernel = KernelSpec(
        kind="lagged_mixed_exponential",
        params={"weights": [[[1.0]]], "beta": [[[1.0]]], "tau": [[[tau]]]},
    )
    with pytest.raises(ValueError, match="tau"):
        simulate_hawkes(
            t_max=1.0,
            baseline=BaselineSpec(kind="constant", params={"values": [1.0]}),
            alpha=[[0.0]],
            kernel=kernel,
            seed=1,
        )


@pytest.mark.parametrize(
    ("name", "tensor"),
    [
        ("weights", []),
        ("weights", [[]]),
        ("weights", [[[]]]),
        ("weights", [[[1.0], [1.0]]]),
        ("weights", [[[1.0]], [[1.0]]]),
        ("beta", []),
        ("beta", [[[1.0, 2.0]]]),
        ("tau", []),
        ("tau", [[[]]]),
        ("tau", [[[0.0, 0.5]]]),
    ],
)
def test_rejects_mismatched_or_empty_shapes(name: str, tensor: list[list[list[float]]]) -> None:
    params = {"weights": [[[1.0]]], "beta": [[[1.0]]], "tau": [[[0.0]]]}
    params[name] = tensor
    with pytest.raises(ValueError, match=r"shape|square|empty|component"):
        simulate_hawkes(
            t_max=1.0,
            baseline=BaselineSpec(kind="constant", params={"values": [1.0]}),
            alpha=[[0.0]],
            kernel=KernelSpec(kind="lagged_mixed_exponential", params=params),
            seed=1,
        )


@pytest.mark.parametrize("kind", ["mixed_exponential", "lagged_mixed_exponential"])
def test_kernel_dimension_must_match_baseline(
    kind: Literal["mixed_exponential", "lagged_mixed_exponential"],
) -> None:
    kernel = KernelSpec(
        kind=kind,
        params={"weights": [[[1.0]]], "beta": [[[1.0]]], "tau": [[[0.0]]]},
    )
    with pytest.raises(ValueError, match="dimension"):
        simulate_hawkes(
            t_max=1.0,
            baseline=BaselineSpec(kind="constant", params={"values": [1.0, 0.0]}),
            alpha=[[0.0, 0.0], [0.0, 0.0]],
            kernel=kernel,
            seed=1,
        )


def test_numpy_parameters_match_nested_sequences() -> None:
    params = {"weights": [[[1.0, 3.0]]], "beta": [[[2.0, 4.0]]], "tau": [[[0.1, 0.5]]]}
    arrays = {name: np.asarray(values) for name, values in params.items()}
    nested = KernelSpec(kind="lagged_mixed_exponential", params=params)
    numpy_kernel = KernelSpec(kind="lagged_mixed_exponential", params=arrays)
    assert numpy_kernel.to_backend() == nested.to_backend()


@pytest.mark.parametrize("alpha", [-1.0, np.nan, np.inf])
def test_rejects_invalid_excitation_even_without_immigrants(alpha: float) -> None:
    kernel = KernelSpec(
        kind="lagged_mixed_exponential",
        params={"weights": [[[1.0]]], "beta": [[[1.0]]], "tau": [[[0.0]]]},
    )
    with pytest.raises(ValueError, match="alpha"):
        simulate_hawkes(
            t_max=1.0,
            baseline=BaselineSpec(kind="constant", params={"values": [0.0]}),
            alpha=[[alpha]],
            kernel=kernel,
            seed=1,
        )


@pytest.mark.parametrize("t_max", [0.0, -1.0, np.nan, np.inf])
def test_rejects_invalid_observation_window(t_max: float) -> None:
    kernel = KernelSpec(
        kind="lagged_mixed_exponential",
        params={"weights": [[[1.0]]], "beta": [[[1.0]]], "tau": [[[0.0]]]},
    )
    with pytest.raises(ValueError, match="t_max"):
        simulate_hawkes(
            t_max=t_max,
            baseline=BaselineSpec(kind="constant", params={"values": [0.0]}),
            alpha=[[0.0]],
            kernel=kernel,
            seed=1,
        )
