$ErrorActionPreference = "Stop"

$PythonVersion = "3.11"

# Create .venv and install dependencies
if (Get-Command uv -ErrorAction SilentlyContinue) {
    Write-Host "Using uv..."
    uv venv --python $PythonVersion .venv
    .\.venv\Scripts\Activate.ps1
    uv pip install -e ".[dev]"
} else {
    Write-Host "Using system python venv..."
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -e ".[dev]"
}

# Install Node deps
npm ci

Write-Host ""
Write-Host "Setup complete."
Write-Host "To start: npm run dev"
