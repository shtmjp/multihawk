"""Python wrapper for the multihawk Rust simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from multihawk import multihawk_rs  # type: ignore[import]

from .spec import BaselineSpec, KernelSpec, _to_float_matrix

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.random import Generator


SamplingMethod = Literal["thinning", "inverse_transform"]


@dataclass
class SimulationResult:
    """Result of Hawkes process simulation."""

    timestamps: list[list[float]]
    obs_window: tuple[float, float]


def simulate_hawkes(
    t_max: float,
    baseline: BaselineSpec,
    alpha: Sequence[Sequence[float]],
    kernel: KernelSpec,
    seed: int | None = None,
    rng: Generator | None = None,
    sampling_method: SamplingMethod = "thinning",
) -> SimulationResult:
    """Simulate a Hawkes process using the unified API."""
    backend_baseline = baseline.to_backend()
    backend_kernel = kernel.to_backend()
    alpha_matrix = _to_float_matrix(alpha)

    if rng is None:
        data = multihawk_rs.simulate_hawkes(
            t_max,
            backend_baseline,
            alpha_matrix,
            backend_kernel,
            seed,
            sampling_method,
        )
    else:
        state: dict[str, Any] = rng.bit_generator.state  # type: ignore[assignment]
        bitgen = state["bit_generator"]
        if bitgen == "PCG64":
            data, state_val, inc = multihawk_rs.simulate_hawkes_pcg64(
                t_max,
                backend_baseline,
                alpha_matrix,
                backend_kernel,
                state["state"]["state"],
                state["state"]["inc"],
                sampling_method,
            )
            state["state"]["state"] = state_val
            state["state"]["inc"] = inc
            rng.bit_generator.state = state
        elif bitgen == "PCG64DXSM":
            data, state_val, inc = multihawk_rs.simulate_hawkes_pcg64dxsm(
                t_max,
                backend_baseline,
                alpha_matrix,
                backend_kernel,
                state["state"]["state"],
                state["state"]["inc"],
                sampling_method,
            )
            state["state"]["state"] = state_val
            state["state"]["inc"] = inc
            rng.bit_generator.state = state
        else:  # pragma: no cover - depends on external RNGs
            msg = f"Unsupported bit generator: {bitgen}"
            raise ValueError(msg)
    return SimulationResult(timestamps=data.timestamps, obs_window=data.obs_window)


__all__ = [
    "SamplingMethod",
    "SimulationResult",
    "simulate_hawkes",
]
