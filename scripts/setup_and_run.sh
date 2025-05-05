#!/bin/bash
# Setup and run script for Llama 4 and Gemini long context comparison

# Stop on error
set -e

# Set variables
VENV_DIR=".venv"
NEEDLE="The secret passphrase for the blueberry muffin recipe is 'QuantumFusionParadox42'."
QUESTION="What is the secret passphrase for the blueberry muffin recipe?"
CHAR_COUNT="10000"  # Start small (~2.5k tokens)
CHUNK_SIZE="1000"   # Process in chunks of this size
GEMINI_API_KEY=${GEMINI_API_KEY:-""}  # Use environment variable or empty
TEST_MODE="yes"     # Default to test mode for safety

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
else
    # Check GPU memory if nvidia-smi is available
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')
    echo "Detected GPU with ${GPU_MEM}MB memory"
    
    if [[ $GPU_MEM -lt 75000 ]]; then
        echo "⚠️ Warning: GPU memory is less than 75GB. This may not be sufficient for full inference."
        echo "The script will run in test mode to validate the setup without full inference."
        TEST_MODE="yes"
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
echo ""
echo "===== Test Configuration ====="
echo "Start with a small context size and gradually increase if successful."
read -p "Enter context size in characters (default: ${CHAR_COUNT}, ~${CHAR_COUNT//[0-9]/}k tokens): " user_char_count
if [ ! -z "$user_char_count" ]; then
    CHAR_COUNT="$user_char_count"
fi

# Ask about chunk size
read -p "Enter chunk size for processing (default: ${CHUNK_SIZE} tokens): " user_chunk_size
if [ ! -z "$user_chunk_size" ]; then
    CHUNK_SIZE="$user_chunk_size"
fi

# Ask about test mode
read -p "Run in test mode? Test mode doesn't generate responses but verifies the model can handle the context. (Y/n): " user_test_mode
if [[ "$user_test_mode" =~ ^[Nn]$ ]]; then
    TEST_MODE="no"
    echo "⚠️ Warning: Running in FULL INFERENCE mode. This requires substantial GPU memory."
else
    TEST_MODE="yes"
    echo "Running in TEST MODE - will verify model loads and can handle context without full inference."
fi

if [[ "$TEST_MODE" == "yes" ]]; then
    TEST_FLAG="--test-mode"
else
    TEST_FLAG=""
fi

# Set CUDA memory management environment variable
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"

# Run the comparison script
echo ""
echo "=== Running Model Comparison ==="
echo "Context size: ${CHAR_COUNT} characters (~${CHAR_COUNT//[0-9]/0}k tokens)"
echo "Chunk size: ${CHUNK_SIZE} tokens"
echo "Test mode: ${TEST_MODE}"
echo "Starting comparison - this may take a while..."
echo ""

# Execute the comparison script
python model_comparison.py --char-count "$CHAR_COUNT" --needle "$NEEDLE" --question "$QUESTION" --chunk-size "$CHUNK_SIZE" $MODEL_FLAG $TEST_FLAG

# Deactivate virtual environment
deactivate

echo ""
echo "Script completed. See above for results summary and check the 'comparison_results' directory for detailed results."

echo ""
echo "Next steps:"
echo "1. If the test was successful, gradually increase the context size"
echo "2. Try with 100,000 characters (~25k tokens) next"
echo "3. Then 1,000,000 characters (~250k tokens)"
echo "4. Then 8,000,000 characters (~2M tokens)"
echo "5. Finally 40,000,000 characters (~10M tokens) for full Llama 4 Scout capacity test" 