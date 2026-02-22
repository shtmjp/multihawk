ARG UV_VERSION=0.8.0
ARG DEBIAN_VERSION=bookworm


FROM ghcr.io/astral-sh/uv:$UV_VERSION AS uv


FROM mcr.microsoft.com/vscode/devcontainers/base:$DEBIAN_VERSION

# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    curl \
    g++ \
    # To remove the image size, it is recommended refresh the package cache as follows
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# install zig
RUN curl -L https://ziglang.org/download/0.15.2/zig-aarch64-linux-0.15.2.tar.xz | tar -xJ -C /opt && ln -sfn /opt/zig-aarch64-linux-0.15.2/zig /usr/local/bin/zig

COPY --from=uv --chown=vscode: /uv /uvx /bin/
USER vscode
# install rust and maturin
# hadolint ignore=SC1091
RUN curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh -s -- -y && . "$HOME/.cargo/env"
RUN uv tool install maturin==1.8.3
ENV PYO3_PYTHON=/workspaces/multihawk/.venv/bin/python
