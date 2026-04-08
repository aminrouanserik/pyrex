FROM python:3.12-slim

# Install system dependencies required by pycbc and scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libfftw3-dev \
    liblapack-dev \
    libblas-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy local source packages (referenced in [tool.uv.sources])
COPY phenomxpy-0.1.0.tar.gz .
COPY qcextender-0.4.9.tar.gz .

# Copy project files
COPY pyproject.toml .
COPY README.md .
COPY src/ ./src/

# Install production dependencies only
RUN uv sync --no-dev

# Make venv binaries available
ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "-m", "pyrex"]