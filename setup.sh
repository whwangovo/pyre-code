#!/bin/bash
set -e

PYTHON_VERSION="3.11"
# CPU-only torch (~200 MB vs ~2 GB of CUDA wheels), matching grading.Dockerfile
TORCH_INDEX="https://download.pytorch.org/whl/cpu"

# Create .venv and install dependencies
if command -v uv &> /dev/null; then
  echo "Using uv..."
  uv venv --python "$PYTHON_VERSION" .venv
  source .venv/bin/activate
  uv pip install torch --index-url "$TORCH_INDEX"
  uv pip install -e ".[dev]"
else
  echo "Using system python venv..."
  python3 -m venv .venv
  source .venv/bin/activate
  pip install torch --index-url "$TORCH_INDEX"
  pip install -e ".[dev]"
fi

# Install Node deps
npm ci

echo ""
echo "Setup complete."
echo "To start: npm run dev"
