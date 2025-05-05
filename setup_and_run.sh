#!/bin/bash
# Setup and run script for Llama 4 and Gemini long context comparison

# Stop on error
set -e

# Set variables
VENV_DIR=".venv"
NEEDLE="The secret passphrase for the blueberry muffin recipe is 'QuantumFusionParadox42'."
QUESTION="What is the secret passphrase for the blueberry muffin recipe?"
CHAR_COUNT="8000000"  # ~2M tokens
GEMINI_API_KEY=${GEMINI_API_KEY:-""}  # Use environment variable or empty

# Print header
echo "=== Llama 4 / Gemini Long Context Test Setup ==="
echo "This script will set up the environment and run the comparison test."

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is required but not found."
    exit 1
fi

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️ Warning: NVIDIA tools not found. CUDA may not be available."
    echo "This script requires a GPU with sufficient VRAM (H100 recommended)."
    read -p "Continue anyway? (y/N): " continue_without_cuda
    if [[ ! "$continue_without_cuda" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Check for Gemini API key
if [ -z "$GEMINI_API_KEY" ]; then
    echo "⚠️ No Gemini API key found in GEMINI_API_KEY environment variable."
    echo "The comparison will only run for Llama 4 unless you provide a key."
    read -p "Would you like to provide a Gemini API key now? (y/N): " provide_key
    if [[ "$provide_key" =~ ^[Yy]$ ]]; then
        read -p "Enter Gemini API key: " GEMINI_API_KEY
    fi
fi

# Determine which models to run
if [ -z "$GEMINI_API_KEY" ]; then
    MODEL_FLAG="--llama-only"
    echo "Will run test on Llama 4 only (no Gemini API key provided)."
else
    MODEL_FLAG="--gemini-api-key $GEMINI_API_KEY"
    echo "Will run test on both Llama 4 and Gemini."
fi

# Ask about context size
read -p "Enter context size in characters (default: ${CHAR_COUNT}, ~2M tokens): " user_char_count
if [ ! -z "$user_char_count" ]; then
    CHAR_COUNT="$user_char_count"
fi

# Run the comparison script
echo ""
echo "=== Running Model Comparison ==="
echo "Context size: ${CHAR_COUNT} characters"
echo "Starting comparison - this may take a while..."
echo ""

# Execute the comparison script
python model_comparison.py --char-count "$CHAR_COUNT" --needle "$NEEDLE" --question "$QUESTION" $MODEL_FLAG

# Deactivate virtual environment
deactivate

echo ""
echo "Script completed. See above for results summary and check the 'comparison_results' directory for detailed results." 