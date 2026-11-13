FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy dependency specifications and readme
COPY pyproject.toml uv.lock README.md ./

# Install dependencies using uv sync without installing project root yet
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source code
COPY app/ ./app/
COPY ui/ ./ui/
COPY eval/ ./eval/

# Install the project itself
RUN uv sync --frozen --no-dev

# Place .venv binaries in PATH
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000 8501

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
