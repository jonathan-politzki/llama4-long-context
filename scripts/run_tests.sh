#!/bin/sh
# Main script to run different long context tests
# This script uses /bin/sh for broader compatibility

# Display help message
show_help() {
  echo "Long Context Testing Suite"
  echo "=========================="
  echo "Usage: $0 [command] [options]"
  echo ""
  echo "Commands:"
  echo "  llama-small     Run a small test with Llama (5K chars)"
  echo "  llama-full      Run the full 400K token test with Llama"
  echo "  gemini          Run a test with Gemini API"
  echo "  gemini-scale    Run a scaling test with Gemini API"
  echo "  compare         Run a comparison between Llama and Gemini"
  echo "  setup           Set up authentication for APIs"
  echo "  docker-build    Build the Docker image"
  echo "  docker-run      Run the Docker container"
  echo ""
  echo "Options:"
  echo "  --size SIZE     Size in characters (default varies by test)"
  echo "  --position POS  Position for the needle (0-100, default: 50)"
  echo ""
  echo "Examples:"
  echo "  $0 llama-small"
  echo "  $0 gemini --size 100000"
  echo "  $0 gemini-scale --max 1600000"
  echo ""
}

# Setup authentication for APIs
setup_auth() {
  echo "Setting up authentication..."
  
  # Run the HuggingFace setup script
  if [ -f "./scripts/setup_huggingface.sh" ]; then
    ./scripts/setup_huggingface.sh
  else
    echo "HuggingFace setup script not found"
  fi
  
  # Check for Gemini API key
  echo ""
  echo "Checking for Gemini API key..."
  if [ -z "$GEMINI_API_KEY" ]; then
    echo "No Gemini API key found in environment"
    echo "Please enter your Gemini API key (or press Enter to skip):"
    read -r api_key
    if [ -n "$api_key" ]; then
      export GEMINI_API_KEY="$api_key"
      echo "Gemini API key set for this session"
      echo "To make it permanent, add this to your shell config:"
      echo "export GEMINI_API_KEY=$api_key"
    else
      echo "Skipping Gemini API setup"
    fi
  else
    echo "Gemini API key found in environment"
  fi
}

# Run a small test with Llama
run_llama_small() {
  echo "Running small Llama test..."
  python3 models/llama/simple_irope_test.py
}

# Run the full Llama test
run_llama_full() {
  size=${1:-1600000}
  echo "Running full Llama test with size $size..."
  python3 models/llama/long_context_test.py
}

# Run a Gemini test
run_gemini() {
  size=100000
  pos=50
  needle=""
  question=""
  
  # Parse command line arguments
  shift
  while [ $# -gt 0 ]; do
    case "$1" in
      --size)
        size="$2"
        shift 2
        ;;
      --position)
        pos="$2"
        shift 2
        ;;
      --needle)
        needle="$2"
        shift 2
        ;;
      --question)
        question="$2"
        shift 2
        ;;
      *)
        echo "Unknown option: $1"
        show_help
        exit 1
        ;;
    esac
  done
  
  # Build command with optional parameters
  cmd="python3 models/gemini/gemini_test.py --char-count $size --position $pos"
  if [ -n "$needle" ]; then
    cmd="$cmd --needle \"$needle\""
  fi
  if [ -n "$question" ]; then
    cmd="$cmd --question \"$question\""
  fi
  
  # Run the command
  echo "Running Gemini test with size $size and position $pos%..."
  if [ -n "$needle" ]; then
    echo "Custom needle: $needle"
  fi
  eval $cmd
}

# Run a Gemini scaling test
run_gemini_scale() {
  max=${1:-8000000}
  echo "Running Gemini scaling test up to $max characters..."
  python3 models/gemini/gemini_test.py --scaling-test --max-chars "$max"
}

# Run a comparison between Llama and Gemini
run_comparison() {
  size=${1:-100000}
  echo "Running comparison with size $size..."
  python3 analysis/model_comparison.py --char-count "$size"
}

# Build the Docker image
build_docker() {
  echo "Building Docker image..."
  docker-compose -f docker/docker-compose.yml build
}

# Run the Docker container
run_docker() {
  echo "Running Docker container..."
  docker-compose -f docker/docker-compose.yml up -d
  docker exec -it llama4-long-context /bin/bash
}

# Main command handling
case "$1" in
  "llama-small")
    run_llama_small
    ;;
  "llama-full")
    run_llama_full "$2"
    ;;
  "gemini")
    run_gemini "$2" "$3" "$4" "$5"
    ;;
  "gemini-scale")
    max=8000000
    
    # Parse command line arguments
    shift
    while [ $# -gt 0 ]; do
      case "$1" in
        --max)
          max="$2"
          shift 2
          ;;
        *)
          echo "Unknown option: $1"
          show_help
          exit 1
          ;;
      esac
    done
    
    run_gemini_scale "$max"
    ;;
  "compare")
    size=100000
    
    # Parse command line arguments
    shift
    while [ $# -gt 0 ]; do
      case "$1" in
        --size)
          size="$2"
          shift 2
          ;;
        *)
          echo "Unknown option: $1"
          show_help
          exit 1
          ;;
      esac
    done
    
    run_comparison "$size"
    ;;
  "setup")
    setup_auth
    ;;
  "docker-build")
    build_docker
    ;;
  "docker-run")
    run_docker
    ;;
  "help"|"-h"|"--help")
    show_help
    ;;
  *)
    echo "Unknown command: $1"
    show_help
    exit 1
    ;;
esac 