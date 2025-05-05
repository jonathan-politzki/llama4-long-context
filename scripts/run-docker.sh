#!/bin/bash
# Helper script to run the Llama 4 tests in Docker

# If no command given, show help
if [ $# -eq 0 ]; then
    echo "Usage: ./run-docker.sh COMMAND [ARGS]"
    echo ""
    echo "Commands:"
    echo "  setup             - Run the interactive setup script"
    echo "  llama             - Run the Llama 4 test only"
    echo "  test SIZE         - Test loading with specified character count"
    echo "  compare SIZE      - Run the comparison with full inference"
    echo "  shell             - Get a shell inside the container"
    echo "  flash-test        - Run a series of tests with Flash Attention 2 enabled"
    echo ""
    echo "Examples:"
    echo "  ./run-docker.sh setup"
    echo "  ./run-docker.sh test 100000"
    echo "  ./run-docker.sh compare 1000000"
    echo "  ./run-docker.sh flash-test"
    exit 0
fi

# Create directories if they don't exist
mkdir -p comparison_results hf_cache

COMMAND=$1
shift

# Build the image if needed
echo "Building Docker image if needed..."
docker compose build

# Run the appropriate command
case $COMMAND in
    setup)
        echo "Running setup script..."
        docker compose run --rm llama-test setup
        ;;
    llama)
        echo "Running Llama 4 test..."
        docker compose run --rm llama-test llama
        ;;
    test)
        SIZE=${1:-10000}
        echo "Testing with character count $SIZE (no inference)..."
        docker compose run --rm llama-test test $SIZE
        ;;
    compare)
        SIZE=${1:-10000}
        echo "Running comparison with character count $SIZE..."
        docker compose run --rm llama-test compare $SIZE
        ;;
    shell)
        echo "Opening shell in container..."
        docker compose run --rm llama-test bash
        ;;
    flash-test)
        echo "=== Running Progressive Context Size Tests with Flash Attention 2 ==="
        echo "This will run a series of tests with increasingly larger contexts"
        echo "to determine the maximum context size achievable for inference."
        echo ""
        
        # Test mode first (loading only)
        echo "Phase 1: Testing context loading (no inference)"
        SIZES=(100000 500000 1000000 2000000 4000000 8000000 12000000 16000000)
        
        for SIZE in "${SIZES[@]}"; do
            echo ""
            echo "Testing context size: $SIZE chars (~$(($SIZE/4)) tokens)"
            docker compose run --rm llama-test test $SIZE
            
            # Check exit status
            if [ $? -ne 0 ]; then
                echo "Failed at context size: $SIZE"
                echo "Maximum successful loading context size: Previous size"
                break
            fi
            
            echo "Success at context size: $SIZE"
        done
        
        echo ""
        echo "Phase 2: Testing inference capability"
        # Start small for inference testing
        INFERENCE_SIZES=(100000 200000 500000 1000000 2000000)
        
        for SIZE in "${INFERENCE_SIZES[@]}"; do
            echo ""
            echo "Testing inference with context size: $SIZE chars (~$(($SIZE/4)) tokens)"
            docker compose run --rm llama-test compare $SIZE
            
            # Check exit status
            if [ $? -ne 0 ]; then
                echo "Failed inference at context size: $SIZE"
                echo "Maximum successful inference context size: Previous size"
                break
            fi
            
            echo "Successful inference at context size: $SIZE"
        done
        
        echo ""
        echo "Test complete. Check the results to determine maximum context size."
        ;;
    *)
        echo "Running custom command: $COMMAND $@"
        docker compose run --rm llama-test $COMMAND "$@"
        ;;
esac 