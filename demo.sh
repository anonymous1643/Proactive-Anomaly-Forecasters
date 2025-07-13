#!/bin/bash

set -e  # Exit immediately if a command fails
echo "Running training script..."
python3 Code/ForwardForecastModel.py

echo "Running evaluation script..."
python3 Code/EvaluateFFM.py

