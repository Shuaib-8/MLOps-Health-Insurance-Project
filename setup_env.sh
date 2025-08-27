#!/bin/bash

# MLOps Health Insurance Project - Environment Setup Script
# This script sets up the conda environment and uv virtual environment

set -e  # Exit on any error

# Function to check if conda environment exists
check_conda_env() {
    conda env list | grep -q "py313-mlops"
}

# Function to create conda environment if it doesn't exist
create_conda_env() {
    echo "🔧 Creating conda environment from environment.yml..."
    if [ -f "environment-minimal.yml" ]; then
        echo "Using environment-minimal.yml (recommended for cross-platform)"
        conda env create -f environment-minimal.yml
    elif [ -f "environment.yml" ]; then
        echo "Using environment.yml (exact versions)"
        conda env create -f environment.yml
    else
        echo "❌ No environment files found. Creating minimal environment..."
        conda create -n py313-mlops python=3.13 llvm-openmp -y
    fi
}

# Function to create uv virtual environment
create_uv_env() {
    echo "🔧 Creating uv virtual environment..."
    # Get the conda environment's Python path
    CONDA_PYTHON=$(conda run -n py313-mlops which python)
    uv venv --python "$CONDA_PYTHON"
    echo "📦 Installing project dependencies with uv..."
    uv sync
}

# Main setup logic
echo "🚀 MLOps Health Insurance Project - Environment Setup"
echo "=================================================="

# Check if conda environment exists
if ! check_conda_env; then
    echo "⚠️  Conda environment 'py313-mlops' not found."
    read -p "Do you want to create it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        create_conda_env
    else
        echo "❌ Setup cancelled."
        exit 1
    fi
else
    echo "✅ Conda environment 'py313-mlops' found."
fi

# Check if uv virtual environment exists
if [ ! -d ".venv" ]; then
    echo "⚠️  uv virtual environment not found."
    read -p "Do you want to create it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        create_uv_env
    else
        echo "❌ Setup cancelled."
        exit 1
    fi
else
    echo "✅ uv virtual environment found."
fi

echo ""
echo "🔄 Activating environments..."
echo "Activating conda environment: py313-mlops"
# Note: conda activate doesn't work in scripts, use conda run or source manually

echo "Activating uv virtual environment"
source .venv/bin/activate

echo ""
echo "✅ Environment setup complete!"
echo "Python version: $(python --version)"
echo "Architecture: $(python -c 'import platform; print(platform.machine())')"
echo "Python path: $(which python)"

# Test key dependencies
echo ""
echo "🧪 Testing key dependencies..."
python -c "import numpy, pandas, xgboost, sklearn; print('✅ All key dependencies working!')" || echo "❌ Some dependencies have issues"

echo ""
echo "🎯 You're ready to work on the MLOps Health Insurance project!"
echo ""
echo "To activate environments manually:"
echo "  conda activate py313-mlops"
echo "  source .venv/bin/activate"
echo ""
echo "Or source this script: source ./activate.sh"
