#!/bin/bash
# Script to run the analysis and visualization of test results

# Create a virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install pandas matplotlib seaborn tabulate

# Create directories
mkdir -p analysis_results

# Run the analysis script
echo "Running analysis..."
python analyze_results.py

# Display results
echo "Analysis complete. Results saved to ./analysis_results/"
echo "To view the results, check the generated images and summary report."

# Deactivate virtual environment
deactivate 