# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

# Install Python 3.10 specifically
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update

# System dependencies
RUN apt-get install -y --no-install-recommends \
    build-essential \
    git \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3.10-venv \
    python3-pip \
    libgl1-mesa-dri \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libglfw3 \
    libglfw3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libx11-6 \
    libxau6 \
    libxdmcp6 \
    mesa-utils \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Install uv with --break-system-packages to bypass PEP 668
RUN python3 -m pip install uv --break-system-packages

# Set workdir
WORKDIR /workspace

# Copy only requirements for cache efficiency
COPY pyproject.toml ./

# Install compatible versions of key libraries first to avoid import errors
RUN uv pip install --system bitsandbytes==0.41.1 triton==2.1.0 transformers==4.35.2 diffusers==0.24.0

# Install project dependencies
RUN uv pip install --system -e .

# Copy the rest of the code
COPY . .

# Remove any existing .venv to avoid conflicts with system packages
RUN rm -rf /workspace/.venv

# Set environment variables
ENV PYTHONPATH=/workspace:$PYTHONPATH
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV DISPLAY=:99
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

# Create display for headless rendering
RUN echo "#!/bin/bash\nXvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &\nexec \"\$@\"" > /usr/local/bin/entrypoint.sh && \
    chmod +x /usr/local/bin/entrypoint.sh

# Default command
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
