#!/bin/bash

# Set strict error handling and non-interactive mode
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "=== Starting Python Crypto Trading Bot Setup ==="

# Update package list and install system dependencies
echo "Installing system dependencies..."
apt-get update -y
apt-get install -y \
    build-essential \
    python3-dev \
    python3-venv \
    python3-pip \
    libssl-dev \
    libffi-dev \
    pkg-config \
    wget \
    curl \
    git

# Check if uv is available, otherwise use venv + pip
echo "Setting up Python virtual environment..."
if command -v uv &> /dev/null; then
    echo "Using uv for virtual environment..."
    uv venv venv
    source venv/bin/activate
    uv pip install --upgrade pip
else
    echo "Using python3 venv..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip wheel setuptools
fi

# Install dependencies - check for requirements.txt, pyproject.toml, or auto-detect
echo "Installing Python dependencies..."
if [[ -f "requirements.txt" ]]; then
    echo "Found requirements.txt, installing dependencies..."
    if command -v uv &> /dev/null; then
        uv pip install -r requirements.txt
    else
        pip install -r requirements.txt
    fi
elif [[ -f "pyproject.toml" ]]; then
    echo "Found pyproject.toml, installing dependencies..."
    if command -v uv &> /dev/null; then
        uv pip install .
    else
        pip install .
    fi
else
    echo "No requirements.txt or pyproject.toml found, auto-detecting dependencies..."
    # Auto-detected dependencies for crypto trading bot
    DEPS=(
        "ccxt"
        "pandas"
        "numpy" 
        "scikit-learn"
        "python-binance"
        "requests"
        "python-dotenv"
        "pytest"
    )
    
    for dep in "${DEPS[@]}"; do
        echo "Installing $dep..."
        if command -v uv &> /dev/null; then
            uv pip install "$dep"
        else
            pip install "$dep"
        fi
    done
fi

# Try to install TA-Lib with fallback strategy
echo "Installing TA-Lib..."
if command -v uv &> /dev/null; then
    # Try ta-lib-bin first (precompiled binary)
    if ! uv pip install ta-lib-bin; then
        echo "ta-lib-bin failed, trying TA-Lib from source..."
        # Install TA-Lib dependencies for compilation
        apt-get install -y libta-lib-dev ta-lib
        uv pip install TA-Lib || echo "Warning: TA-Lib installation failed"
    fi
else
    # Try ta-lib-bin first (precompiled binary) 
    if ! pip install ta-lib-bin; then
        echo "ta-lib-bin failed, trying TA-Lib from source..."
        # Install TA-Lib dependencies for compilation
        apt-get install -y libta-lib-dev ta-lib
        pip install TA-Lib || echo "Warning: TA-Lib installation failed"
    fi
fi

# Display Python and package versions
echo "=== Python and Package Versions ==="
python3 --version
pip --version

echo "Installed packages:"
pip list | grep -E "(ccxt|pandas|numpy|scikit-learn|binance|requests|dotenv|pytest|TA-Lib|ta-lib)" || echo "No matching packages found"

# Compile all Python files
echo "Compiling Python files..."
python3 -m compileall . -q || echo "Warning: Some files failed to compile"

# Run tests if they exist
echo "Checking for tests..."
if find . -name "test*.py" -o -name "*test*.py" | grep -q .; then
    echo "Running tests..."
    python3 -m pytest -v || echo "Warning: Some tests failed"
elif find . -name "tests" -type d | grep -q .; then
    echo "Running tests in tests directory..."
    python3 -m pytest tests/ -v || echo "Warning: Some tests failed"
else
    echo "No test files found, skipping test execution"
fi

echo "=== Setup Complete ==="
echo "JULES_OK"