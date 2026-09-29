FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.15 /uv /uvx /bin/

# PyTorch build: cu126 (NVIDIA GPU, run with --gpus all) or cpu
ARG TORCH=cu126
# Recorded in each training's run_info.json, since the image has no .git
ARG GIT_COMMIT=unknown
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    DEEPANESEG_GIT_COMMIT=${GIT_COMMIT} \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies first, so that code changes don't reinstall them
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev --extra ${TORCH} && rm -rf /root/.cache/uv

COPY . .
RUN uv sync --locked --no-dev --extra ${TORCH} && rm -rf /root/.cache/uv

CMD ["deepaneseg-train", "--help"]
