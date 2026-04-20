@echo off
setlocal

set PYTHON_VERSION=3.11

where uv >nul 2>nul
if %errorlevel% equ 0 (
    echo Using uv...
    uv venv --python %PYTHON_VERSION% .venv
    call .venv\Scripts\activate.bat
    uv pip install -e ".[dev]"
) else (
    echo Using system python venv...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    pip install -e ".[dev]"
)

rem Install Node deps
npm ci

echo.
echo Setup complete.
echo To start: npm run dev
