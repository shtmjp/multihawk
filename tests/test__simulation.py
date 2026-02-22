from __future__ import annotations

import numpy as np
import pytest

from multihawk import BaselineSpec, KernelSpec, simulate_hawkes


def test_simulate_hawkes_exponential_reproducible() -> None:
    """Check reproducibility and bounds for exponential kernel."""
    baseline = BaselineSpec(kind="constant", params={"values": [0.2, 0.1]})
    kernel = KernelSpec(kind="exponential", params={"beta": [[2.0, 3.0], [1.0, 2.5]]})
    alpha = [[0.2, 0.1], [0.0, 0.1]]
    t_max = 100.0
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    result1 = simulate_hawkes(t_max=t_max, baseline=baseline, alpha=alpha, kernel=kernel, rng=rng1)
    result2 = simulate_hawkes(t_max=t_max, baseline=baseline, alpha=alpha, kernel=kernel, rng=rng2)
    assert result1.obs_window == (0.0, t_max)
    assert result1.timestamps == result2.timestamps
    assert len(result1.timestamps) == len(baseline.params["values"])
    assert all(all(0.0 <= t <= t_max for t in ts) for ts in result1.timestamps)


def test_simulate_hawkes_inverse_transform_sampling() -> None:
    """Check inverse-transform immigrant sampling outputs and ranges."""
    baseline = BaselineSpec(kind="constant", params={"values": [0.2, 0.1]})
    kernel = KernelSpec(kind="exponential", params={"beta": [[1.5, 0.5], [0.3, 1.2]]})
    alpha = [[0.1, 0.0], [0.0, 0.05]]
    t_max = 50.0
    result = simulate_hawkes(
        t_max=t_max,
        baseline=baseline,
        alpha=alpha,
        kernel=kernel,
        rng=np.random.default_rng(321),
        sampling_method="inverse_transform",
    )
    assert result.obs_window == (0.0, t_max)
    assert len(result.timestamps) == len(baseline.params["values"])
    assert all(all(0.0 <= t <= t_max for t in ts) for ts in result.timestamps)


def test_simulate_hawkes_gamma_reproducible() -> None:
    """Check reproducibility for gamma kernel."""
    baseline = BaselineSpec(kind="constant", params={"values": [0.2, 0.1]})
    kernel = KernelSpec(
        kind="gamma",
        params={"shape": [[2.0, 3.0], [1.0, 2.5]], "rate": [[1.5, 2.0], [2.0, 1.0]]},
    )
    alpha = [[0.2, 0.1], [0.0, 0.1]]
    t_max = 100.0
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    result1 = simulate_hawkes(t_max=t_max, baseline=baseline, alpha=alpha, kernel=kernel, rng=rng1)
    result2 = simulate_hawkes(t_max=t_max, baseline=baseline, alpha=alpha, kernel=kernel, rng=rng2)
    assert result1.obs_window == (0.0, t_max)
    assert result1.timestamps == result2.timestamps
    assert len(result1.timestamps) == len(baseline.params["values"])
    assert all(all(0.0 <= t <= t_max for t in ts) for ts in result1.timestamps)


def test_invalid_sampling_method_raises() -> None:
    """Passing an unsupported `sampling_method` raises ValueError (validation)."""
    baseline = BaselineSpec(kind="constant", params={"values": [0.2]})
    kernel = KernelSpec(kind="exponential", params={"beta": [[1.0]]})
    with pytest.raises(ValueError, match="unsupported immigrant sampling method"):
        simulate_hawkes(
            t_max=10.0,
            baseline=baseline,
            alpha=[[0.0]],
            kernel=kernel,
            rng=np.random.default_rng(1),
            sampling_method="invalid",  # type: ignore[]
        )


def test_simulate_hawkes_mixed_exponential_reproducible() -> None:
    """Check reproducibility for mixed-exponential kernel."""
    baseline = BaselineSpec(kind="constant", params={"values": [0.2, 0.1]})
    weights = [
        [[0.7, 0.3], [0.4, 0.6]],
        [[0.5, 0.5], [0.2, 0.8]],
    ]
    beta = [
        [[1.5, 3.0], [0.5, 1.5]],
        [[1.0, 2.5], [0.8, 2.0]],
    ]
    kernel = KernelSpec(
        kind="mixed_exponential",
        params={"weights": weights, "beta": beta},
    )
    alpha = [[0.2, 0.1], [0.0, 0.1]]
    t_max = 100.0
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    result1 = simulate_hawkes(t_max=t_max, baseline=baseline, alpha=alpha, kernel=kernel, rng=rng1)
    result2 = simulate_hawkes(t_max=t_max, baseline=baseline, alpha=alpha, kernel=kernel, rng=rng2)
    assert result1.obs_window == (0.0, t_max)
    assert result1.timestamps == result2.timestamps
    assert len(result1.timestamps) == len(baseline.params["values"])
    assert all(all(0.0 <= t <= t_max for t in ts) for ts in result1.timestamps)


def test_simulate_hawkes_power_law_reproducible() -> None:
    """Power-law aliases are equivalent; reproducibility holds."""
    baseline = BaselineSpec(kind="constant", params={"values": [0.2, 0.1]})
    delta = [[0.5, 1.0], [1.5, 0.8]]
    beta = [[2.5, 2.1], [3.0, 1.8]]
    kernel = KernelSpec(kind="power_law", params={"delta": delta, "beta": beta})
    alpha = [[0.2, 0.1], [0.05, 0.1]]
    t_max = 100.0
    result1 = simulate_hawkes(
        t_max=t_max,
        baseline=baseline,
        alpha=alpha,
        kernel=kernel,
        rng=np.random.default_rng(123),
    )
    kernel_alias = KernelSpec(
        kind="power_law",
        params={"cutoff": delta, "exponent": beta},
    )
    result2 = simulate_hawkes(
        t_max=t_max,
        baseline=baseline,
        alpha=alpha,
        kernel=kernel_alias,
        rng=np.random.default_rng(123),
    )
    assert result1.obs_window == (0.0, t_max)
    assert result1.timestamps == result2.timestamps
    assert len(result1.timestamps) == len(baseline.params["values"])
    assert all(all(0.0 <= t <= t_max for t in ts) for ts in result1.timestamps)


def test_simulate_hawkes_sequence_baseline_matches_spec() -> None:
    """Same BaselineSpec and seed yield identical timestamps."""
    baseline_spec = BaselineSpec(kind="constant", params={"values": [0.3, 0.2]})
    kernel = KernelSpec(kind="exponential", params={"beta": [[1.5, 0.5], [0.2, 1.0]]})
    alpha = [[0.1, 0.05], [0.0, 0.02]]
    rng1 = np.random.default_rng(99)
    rng2 = np.random.default_rng(99)
    result_spec = simulate_hawkes(
        t_max=25.0, baseline=baseline_spec, alpha=alpha, kernel=kernel, rng=rng1
    )
    # Interface now requires BaselineSpec; verify reproducibility with same Spec
    result_seq = simulate_hawkes(
        t_max=25.0, baseline=baseline_spec, alpha=alpha, kernel=kernel, rng=rng2
    )
    assert result_spec.timestamps == result_seq.timestamps


def test_piecewise_constant_baseline_truncation() -> None:
    """Piecewise-constant baseline: no events after last breakpoint."""
    baseline = BaselineSpec(
        kind="piecewise_constant",
        params={"breaks": [0.0, 2.0, 5.0], "rates": [[0.3, 0.0]]},
    )
    kernel = KernelSpec(kind="exponential", params={"beta": [[1.0]]})
    result = simulate_hawkes(
        t_max=7.5, baseline=baseline, alpha=[[0.0]], kernel=kernel, rng=np.random.default_rng(11)
    )
    assert result.obs_window == (0.0, 7.5)
    assert len(result.timestamps) == 1
    assert all(t <= 5.0 + 1e-12 for t in result.timestamps[0])


def test_piecewise_linear_baseline_truncation() -> None:
    """Piecewise-linear baseline: no events after last breakpoint."""
    baseline = BaselineSpec(
        kind="piecewise_linear",
        params={"breaks": [0.0, 2.0, 5.0], "values": [[0.2, 0.3, 0.0]]},
    )
    kernel = KernelSpec(kind="exponential", params={"beta": [[1.0]]})
    result = simulate_hawkes(
        t_max=7.5, baseline=baseline, alpha=[[0.0]], kernel=kernel, rng=np.random.default_rng(21)
    )
    assert result.obs_window == (0.0, 7.5)
    assert len(result.timestamps) == 1
    assert all(t <= 5.0 + 1e-12 for t in result.timestamps[0])


def test_piecewise_linear_baseline_python_validation() -> None:
    """Missing values triggers Python-side ValueError."""
    baseline = BaselineSpec(kind="piecewise_linear", params={"breaks": [0.0, 1.0]})
    kernel = KernelSpec(kind="exponential", params={"beta": [[1.0]]})
    with pytest.raises(ValueError, match="requires 'breaks' and 'values'"):
        simulate_hawkes(
            t_max=1.0, baseline=baseline, alpha=[[0.0]], kernel=kernel, rng=np.random.default_rng(0)
        )


def test_piecewise_linear_baseline_rust_validation() -> None:
    """Non-increasing breaks trigger Rust-side ValueError."""
    baseline = BaselineSpec(
        kind="piecewise_linear",
        params={"breaks": [0.0, 1.0, 0.5], "values": [[0.1, 0.2, 0.3]]},
    )
    kernel = KernelSpec(kind="exponential", params={"beta": [[1.0]]})
    with pytest.raises(ValueError, match="baseline breakpoints must be strictly increasing"):
        simulate_hawkes(
            t_max=1.0, baseline=baseline, alpha=[[0.0]], kernel=kernel, rng=np.random.default_rng(0)
        )


@pytest.mark.parametrize("bitgen", [np.random.PCG64, np.random.PCG64DXSM])
def test_simulate_hawkes_rng_state(bitgen: type[np.random.BitGenerator]) -> None:
    """Same BitGenerator+seed produce identical timestamps and RNG state."""
    baseline = BaselineSpec(kind="constant", params={"values": [0.2, 0.1]})
    kernel = KernelSpec(kind="exponential", params={"beta": [[2.0, 3.0], [1.0, 2.5]]})
    alpha = [[0.2, 0.1], [0.0, 0.1]]
    rng1 = np.random.Generator(bitgen(123))
    rng2 = np.random.Generator(bitgen(123))
    result1 = simulate_hawkes(t_max=100.0, baseline=baseline, alpha=alpha, kernel=kernel, rng=rng1)
    result2 = simulate_hawkes(t_max=100.0, baseline=baseline, alpha=alpha, kernel=kernel, rng=rng2)
    assert result1.timestamps == result2.timestamps
    assert rng1.bit_generator.state == rng2.bit_generator.state


@pytest.mark.parametrize("bitgen", [np.random.PCG64, np.random.PCG64DXSM])
def test_simulate_hawkes_gamma_rng_state(bitgen: type[np.random.BitGenerator]) -> None:
    """Gamma kernel: same BitGenerator+seed match timestamps and RNG state."""
    baseline = BaselineSpec(kind="constant", params={"values": [0.2, 0.1]})
    kernel = KernelSpec(
        kind="gamma",
        params={"shape": [[2.0, 3.0], [1.0, 2.5]], "rate": [[1.5, 2.0], [2.0, 1.0]]},
    )
    alpha = [[0.2, 0.1], [0.0, 0.1]]
    rng1 = np.random.Generator(bitgen(123))
    rng2 = np.random.Generator(bitgen(123))
    result1 = simulate_hawkes(t_max=100.0, baseline=baseline, alpha=alpha, kernel=kernel, rng=rng1)
    result2 = simulate_hawkes(t_max=100.0, baseline=baseline, alpha=alpha, kernel=kernel, rng=rng2)
    assert result1.timestamps == result2.timestamps
    assert rng1.bit_generator.state == rng2.bit_generator.state


@pytest.mark.parametrize("bitgen", [np.random.PCG64, np.random.PCG64DXSM])
def test_rng_state_advancement(bitgen: type[np.random.BitGenerator]) -> None:
    """RNG state advances; restoring state reproduces results."""
    baseline = BaselineSpec(kind="constant", params={"values": [0.1, 0.1]})
    kernel = KernelSpec(kind="exponential", params={"beta": [[1.0, 1.0], [1.0, 1.0]]})
    alpha = [[0.1, 0.1], [0.1, 0.1]]
    rng = np.random.Generator(bitgen(123))
    state_before = rng.bit_generator.state["state"]["state"]
    simulate_hawkes(t_max=10.0, baseline=baseline, alpha=alpha, kernel=kernel, rng=rng)
    state_after = rng.bit_generator.state["state"]["state"]
    assert state_before != state_after
    rng_cont = np.random.Generator(bitgen())
    rng_cont.bit_generator.state = rng.bit_generator.state
    res1 = simulate_hawkes(t_max=10.0, baseline=baseline, alpha=alpha, kernel=kernel, rng=rng)
    res2 = simulate_hawkes(t_max=10.0, baseline=baseline, alpha=alpha, kernel=kernel, rng=rng_cont)
    assert res1.timestamps == res2.timestamps
