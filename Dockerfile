# syntax=docker/dockerfile:1
FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --upgrade pip && pip install uv

# Set workdir
WORKDIR /workspace

# Copy only requirements for cache efficiency
COPY pyproject.toml ./

# Install compatible versions of key libraries first to avoid import errors
RUN uv pip install --system bitsandbytes==0.41.1 triton==2.1.0 transformers==4.35.2 diffusers==0.24.0

# Install project dependencies
RUN uv pip install --system -e .

# (Optional) For training extras and flash-attn
# RUN uv pip install --system -e .[train] && uv pip install --system flash-attn --no-build-isolation

# Copy the rest of the code
COPY . .

# Default command
CMD ["bash"]
