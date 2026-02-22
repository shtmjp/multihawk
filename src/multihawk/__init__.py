"""Python interface for the multihawk simulator."""

from multihawk import multihawk_rs  # type: ignore[import]

# Local estimator module lives in this package
from .simulation import SamplingMethod, SimulationResult, simulate_hawkes
from .spec import (
    BaselineSpec,
    KernelSpec,
)

__all__ = [
    "BaselineSpec",
    "KernelSpec",
    "SamplingMethod",
    "SimulationResult",
    "multihawk_rs",  # expose for advanced use cases
    "simulate_hawkes",
]
