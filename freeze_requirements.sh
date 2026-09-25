#!/bin/bash

echo "📦 Generating requirements.txt from current environment..."

# Activate venv if it exists
if [ -d "psdenv" ]; then
    echo "🔧 Activating virtual environment..."
    source psdenv/bin/activate
fi

# Generate requirements.txt
pip freeze > requirements.txt

echo "✅ requirements.txt created successfully!"
