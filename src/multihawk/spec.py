"""Specification objects shared across the multihawk Python API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any, Literal


def _to_float_list(seq: Sequence[float]) -> list[float]:
    return [float(v) for v in seq]


def _to_float_matrix(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    return [_to_float_list(row) for row in matrix]


def _to_float_tensor(
    tensor: Sequence[Sequence[Sequence[float]]],
) -> list[list[list[float]]]:
    return [_to_float_matrix(matrix) for matrix in tensor]


def _exponential_kernel_backend(
    kind: Literal["exponential", "lagged_exponential"],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    if "beta" not in params:
        if kind == "exponential":
            msg = "KernelSpec for 'exponential' requires 'beta'"
        else:
            msg = "KernelSpec for 'lagged_exponential' requires 'beta' and 'tau'"
        raise ValueError(msg)

    beta = params["beta"]
    if kind == "exponential":
        return {"kind": kind, "params": {"beta": _to_float_matrix(beta)}}

    if "tau" not in params:
        msg = "KernelSpec for 'lagged_exponential' requires 'beta' and 'tau'"
        raise ValueError(msg)
    tau = params["tau"]
    return {
        "kind": kind,
        "params": {
            "beta": _to_float_matrix(beta),
            "tau": _to_float_matrix(tau),
        },
    }


@dataclass
class BaselineSpec:
    """Specification for non-homogeneous baseline intensities."""

    kind: Literal["constant", "piecewise_constant", "piecewise_linear"]
    params: Mapping[str, Any]

    def to_backend(self) -> dict[str, Any]:
        """Convert to a dictionary suitable for the Rust backend."""
        if self.kind == "constant":
            if "values" not in self.params:
                msg = "BaselineSpec for 'constant' requires a 'values' entry"
                raise ValueError(msg)
            values = self.params["values"]
            return {"kind": self.kind, "params": {"values": _to_float_list(values)}}

        if self.kind == "piecewise_constant":
            if "breaks" not in self.params or "rates" not in self.params:
                msg = "BaselineSpec for 'piecewise_constant' requires 'breaks' and 'rates'"
                raise ValueError(msg)
            breaks = self.params["breaks"]
            rates = self.params["rates"]
            return {
                "kind": self.kind,
                "params": {
                    "breaks": _to_float_list(breaks),
                    "rates": _to_float_matrix(rates),
                },
            }

        if self.kind == "piecewise_linear":
            if "breaks" not in self.params or "values" not in self.params:
                msg = "BaselineSpec for 'piecewise_linear' requires 'breaks' and 'values'"
                raise ValueError(msg)
            breaks = self.params["breaks"]
            values = self.params["values"]
            return {
                "kind": self.kind,
                "params": {
                    "breaks": _to_float_list(breaks),
                    "values": _to_float_matrix(values),
                },
            }

        msg = f"Unsupported baseline kind: {self.kind}"
        raise ValueError(msg)


@dataclass
class KernelSpec:
    """Specification for triggering kernels."""

    kind: Literal[
        "exponential",
        "lagged_exponential",
        "gamma",
        "mixed_exponential",
        "power_law",
    ]
    params: Mapping[str, Any]

    def to_backend(self) -> dict[str, Any]:
        """Convert to a dictionary suitable for the Rust backend."""
        if self.kind in {"exponential", "lagged_exponential"}:
            return _exponential_kernel_backend(self.kind, self.params)

        if self.kind == "gamma":
            if "shape" not in self.params or "rate" not in self.params:
                msg = "KernelSpec for 'gamma' requires 'shape' and 'rate'"
                raise ValueError(msg)
            shape = self.params["shape"]
            rate = self.params["rate"]
            return {
                "kind": self.kind,
                "params": {
                    "shape": _to_float_matrix(shape),
                    "rate": _to_float_matrix(rate),
                },
            }

        if self.kind == "mixed_exponential":
            if "weights" not in self.params or "beta" not in self.params:
                msg = "KernelSpec for 'mixed_exponential' requires 'weights' and 'beta'"
                raise ValueError(msg)
            weights = self.params["weights"]
            beta = self.params["beta"]
            return {
                "kind": self.kind,
                "params": {
                    "weights": _to_float_tensor(weights),
                    "beta": _to_float_tensor(beta),
                },
            }

        if self.kind == "power_law":
            delta = self.params.get("delta", self.params.get("cutoff"))
            beta = self.params.get("beta", self.params.get("exponent"))
            if delta is None or beta is None:
                msg = "KernelSpec 'power_law' requires 'delta/cutoff' and 'beta/exponent'"
                raise ValueError(msg)
            return {
                "kind": self.kind,
                "params": {
                    "delta": _to_float_matrix(delta),
                    "beta": _to_float_matrix(beta),
                },
            }

        msg = f"Unsupported kernel kind: {self.kind}"
        raise ValueError(msg)


__all__ = [
    "BaselineSpec",
    "KernelSpec",
    "_to_float_matrix",
]
