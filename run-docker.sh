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
    echo ""
    echo "Examples:"
    echo "  ./run-docker.sh setup"
    echo "  ./run-docker.sh test 100000"
    echo "  ./run-docker.sh compare 1000000"
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
    *)
        echo "Running custom command: $COMMAND $@"
        docker compose run --rm llama-test $COMMAND "$@"
        ;;
esac 