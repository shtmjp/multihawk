# MultiHawk: Fast simulation for multivariate Hawkes processes with time-varying baseline functions


```python
import numpy as np

from multihawk import BaselineSpec, KernelSpec, simulate_hawkes

rng = np.random.default_rng(0)
baseline = BaselineSpec(
    kind="piecewise_linear",
    params={
        "breaks": [0.0, 10.0, 20.0],
        "values": [[0.1, 0.3, 0.05], [0.05, 0.2, 0.0]],
    },
)
kernel = KernelSpec(
    kind="mixed_exponential",
    params={
        "weights": [[[0.7, 0.3], [0.4, 0.6]], [[0.5, 0.5], [0.2, 0.8]]],
        "beta": [[[1.5, 3.0], [0.5, 1.5]], [[1.0, 2.5], [0.8, 2.0]]],
    },
)
alpha = [[0.2, 0.1], [0.0, 0.1]]

result = simulate_hawkes(
    t_max=50.0,
    baseline=baseline,
    alpha=alpha,
    kernel=kernel,
    rng=rng,
)

isinstance(result.timestamps, list)  # True, list of dimensions
isinstance(result.timestamps[0], list)  # True, first dimension events
isinstance(result.timestamps[0][0], float)  # True, first event time of first dimension

```

The Python API centers on `simulate_hawkes`, which accepts lightweight
specifications for both the baseline intensity and triggering kernel.  The
function accepts an optional `rng` argument which can be a
``numpy.random.Generator``. When supplied, the underlying PCG64 or PCG64DXSM
state is forwarded to the Rust backend so that two generators with identical
state yield identical simulations.  If no generator is given, a simple integer
``seed`` can be used instead.

## Installation
`uv add git+https://github.com/shtmjp/multihawk.git`

### Currently supported specifications

Baseline specifications accept the following `kind` values:

- `"constant"`: a time-homogeneous baseline intensity.
- `"piecewise_constant"`: segment-specific intensities defined by breakpoints.
- `"piecewise_linear"`: linear interpolation between breakpoint intensities.

Kernel specifications currently support:

- `"exponential"`: exponential decay per interaction pair.
- `"lagged_exponential"`: exponential decay beginning after a deterministic
  non-negative lag for each interaction pair.
- `"gamma"`: gamma-distributed triggering with configurable shape and rate.
- `"mixed_exponential"`: mixtures of exponential components for each pair.

  More families can be added by extending the spec objects.


## Example usage

Simulate a two-dimensional stationary Hawkes process with an exponential kernel:

```python
import numpy as np

from multihawk import BaselineSpec, KernelSpec, simulate_hawkes

rng = np.random.default_rng(0)
baseline = BaselineSpec(kind="constant", params={"values": [0.2, 0.1]})
kernel = KernelSpec(kind="exponential", params={"beta": [[2.0, 3.0], [1.0, 2.5]]})
alpha = [[0.2, 0.1], [0.0, 0.1]]

result_exp = simulate_hawkes(t_max=50.0, baseline=baseline, alpha=alpha, kernel=kernel, rng=rng)
print(result_exp.timestamps)
```

For a lagged exponential kernel, `tau[parent][child]` is the deterministic
delay before the exponential triggering density begins. Matrix entries in
`alpha`, `beta`, and `tau` all use the `[parent][child]` convention. For a
source/parent component `j` triggering a target/child component `i`:

\[
\phi_{j\to i}(u)
= \alpha_{ji}\,\beta_{ji}
  \exp\{-\beta_{ji}(u-\tau_{ji})\}
  \mathbf{1}\{u>\tau_{ji}\}.
\]

```python
kernel = KernelSpec(
    kind="lagged_exponential",
    params={
        "beta": [[2.0, 3.0], [1.0, 2.5]],
        "tau": [[0.10, 0.35], [0.25, 0.15]],
    },
)

result_lagged = simulate_hawkes(
    t_max=50.0,
    baseline=baseline,
    alpha=alpha,
    kernel=kernel,
    rng=np.random.default_rng(0),
)
```

For a gamma kernel, adjust the kernel specification:

```python
import numpy as np

from multihawk import BaselineSpec, KernelSpec, simulate_hawkes

rng = np.random.default_rng(0)
baseline = BaselineSpec(kind="constant", params={"values": [0.2, 0.1]})
kernel = KernelSpec(
    kind="gamma",
    params={"shape": [[2.0, 3.0], [1.0, 2.5]], "rate": [[1.5, 2.0], [2.0, 1.0]]},
)
alpha = [[0.2, 0.1], [0.0, 0.1]]

result_gamma = simulate_hawkes(t_max=50.0, baseline=baseline, alpha=alpha, kernel=kernel, rng=rng)
```

To mix multiple exponential components for each interaction, use the
``mixed_exponential`` kernel. The ``weights`` parameter supplies the mixture
weights while ``beta`` provides the corresponding rates:

```python
import numpy as np

from multihawk import BaselineSpec, KernelSpec, simulate_hawkes

rng = np.random.default_rng(0)
baseline = BaselineSpec(kind="constant", params={"values": [0.2, 0.1]})
kernel = KernelSpec(
    kind="mixed_exponential",
    params={
        "weights": [[[0.7, 0.3], [0.4, 0.6]], [[0.5, 0.5], [0.2, 0.8]]],
        "beta": [[[1.5, 3.0], [0.5, 1.5]], [[1.0, 2.5], [0.8, 2.0]]],
    },
)
alpha = [[0.2, 0.1], [0.0, 0.1]]

result_mixed = simulate_hawkes(
    t_max=50.0,
    baseline=baseline,
    alpha=alpha,
    kernel=kernel,
    rng=rng,
)
```

To simulate with a time-varying baseline, build the appropriate `BaselineSpec`:

```python
import numpy as np

from multihawk import BaselineSpec, KernelSpec, simulate_hawkes

rng = np.random.default_rng(0)
baseline = BaselineSpec(
    kind="piecewise_constant",
    params={
        "breaks": [0.0, 10.0, 20.0],
        "rates": [[0.1, 0.3], [0.05, 0.2]],
    },
)

kernel = KernelSpec(kind="exponential", params={"beta": [[2.0, 3.0], [1.5, 2.0]]})

result = simulate_hawkes(
    t_max=30.0,
    baseline=baseline,
    alpha=[[0.2, 0.1], [0.05, 0.1]],
    kernel=kernel,
    rng=rng,
)
```

Piecewise-linear baselines describe a linear interpolation between knots at the
specified breakpoints. Supply the intensity at each knot for every dimension:

```python
import numpy as np

from multihawk import BaselineSpec, KernelSpec, simulate_hawkes

rng = np.random.default_rng(0)
baseline = BaselineSpec(
    kind="piecewise_linear",
    params={
        "breaks": [0.0, 10.0, 20.0],
        "values": [[0.1, 0.3, 0.05], [0.05, 0.2, 0.0]],
    },
)

kernel = KernelSpec(kind="exponential", params={"beta": [[2.0, 3.0], [1.5, 2.0]]})

result_linear = simulate_hawkes(
    t_max=30.0,
    baseline=baseline,
    alpha=[[0.2, 0.1], [0.05, 0.1]],
    kernel=kernel,
    rng=rng,
)
```


## Related Python libraries

- **tick** — Comprehensive inference & simulation (but may be difficult to install currently?)
- **HawkesPyLib** — Lightweight univariate simulation (Ogata thinning) and MLE; accelerated with Numba; exponential/sum-of-exponentials/approx. power-law kernels.
- **omitakahiro/hawkes** — Simulation & MLE; kernels: exponential, multi-exponential, power-law, nonparametric; baselines: constant/piecewise-constant/piecewise-linear/log-linear/custom.
- **hawkeslib** — Fast parameter estimation for vanilla Hawkes (Cython).
- **hawkesbook** — Educational implementations accompanying the book; simple estimation & simulation.
- **stmorse/hawkes** — Minimal multivariate Hawkes for learning/testing.
- **Sparklen** — High-dimensional exponential Hawkes with a C++ core and regularization.

**Positioning** — MultiHawk focuses on fast, reproducible multivariate simulation with various specifications: `BaselineSpec` supports constant/piecewise-constant/piecewise-linear, and `KernelSpec` supports exponential/gamma/mixed-exponential. It forwards `numpy.random.Generator` (PCG64/PCG64DXSM) state to a Rust backend to guarantee bitwise-reproducible runs. Use MultiHawk for large-scale synthetic data and benchmarking, and combine with `tick`/`pyhawkes`/others for parameter estimation on simulated or real data.

## Development notes
- Release:
`maturin build --release -i python3.13 --zig --target aarch64-unknown-linux-gnu -o dist && uv publish`

- Further development plans:
  - Add likelihoods written in JAX for easy gradient-based optimization
  - More kernels and baselines (suggestions are welcome)
