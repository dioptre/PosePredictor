#!/bin/bash
# Setup script for pixels2pose project

echo "Setting up pixels2pose environment..."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv .venv --python 3.8
fi

# Activate and install dependencies
echo "Installing dependencies..."
source .venv/bin/activate
uv pip install -r requirements.txt

echo "Setup complete! To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To run the project:"
echo "  python Pixel2Pose.py --scenario 1"