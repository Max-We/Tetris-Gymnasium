# Start with the latest NVIDIA CUDA base image
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
  ffmpeg git vim curl software-properties-common \
  libglew-dev x11-xserver-utils xvfb


# Set work directory
WORKDIR /app

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

# Configure Poetry to not create virtual environments
RUN poetry config virtualenvs.create false

COPY .dockerignore .

# Copy poetry.lock* in case it doesn't exist in the repo
COPY ./pyproject.toml ./poetry.lock* ./

# Install dependencies
RUN poetry install --no-root --no-interaction

ENV PYTHONPATH="${PYTHONPATH}:/app"

# Copy project files
COPY . .

# Install project
RUN poetry install --no-root --no-interaction --with examples

# Make examples executable
RUN chmod -R +x examples
