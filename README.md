# Long Context Testing Suite

**Version:** 1.0.0

A comprehensive testing framework for evaluating the long-context capabilities of large language models, with focus on:

1. **iRoPE Implementation** for Llama models (interleaved Rotary Position Embeddings)
2. **Gemini API Integration** for comparative testing
3. **Needle-in-Haystack Methodology** for rigorous context evaluation

## Repository Structure

```
llama4-long-context/
├── docker/                 # Docker configuration files
├── scripts/                # Shell scripts for setup and running tests
├── models/
│   ├── llama/              # Llama model tests (400K token context)
│   └── gemini/             # Gemini API integration (2M token context)
├── utils/                  # Shared utility functions
├── analysis/               # Analysis and visualization tools
├── results/                # Test result storage
└── Documentation/          # Detailed implementation documentation
```

## Setup & Installation

### Option 1: Local Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup API authentication**:
   ```bash
   ./scripts/run_tests.sh setup
   ```

3. **Verify installation**:
   ```bash
   ./scripts/run_tests.sh llama-small
   ```

### Option 2: Docker Setup (Recommended for Lambda)

1. **Build the Docker image**:
   ```bash
   ./scripts/run_tests.sh docker-build
   ```

2. **Run the Docker container**:
   ```bash
   ./scripts/run_tests.sh docker-run
   ```

3. **Inside the container, run setup**:
   ```bash
   ./scripts/run_tests.sh setup
   ```

## Running Tests

### Llama Model Tests

Test Llama 4 Scout with iRoPE for long context:

```bash
# Simple test with small context
./scripts/run_tests.sh llama-small

# Full test with 400K tokens
./scripts/run_tests.sh llama-full
```

### Gemini API Tests

Test Google's Gemini 1.5 Pro with 2M token contexts:

```bash
# Run with 100K characters (~25K tokens)
./scripts/run_tests.sh gemini --size 100000

# Run scalability test with multiple sizes
./scripts/run_tests.sh gemini-scale --max 4000000
```

### Comparison Tests

Compare Llama and Gemini on the same task:

```bash
./scripts/run_tests.sh compare --size 100000
```

## Key Features

1. **Needle-in-Haystack Methodology**: Rigorously tests a model's ability to locate specific information in a long context.

2. **iRoPE Implementation**: Implements Meta's interleaved Rotary Position Embeddings to enhance context length capabilities.

3. **Scalability Testing**: Evaluates model performance across different context lengths.

4. **Memory Optimization**: Implements techniques like 4-bit quantization and Flash Attention 2 to maximize context length.

5. **Comprehensive Metrics**: Tracks success rates, response times, and memory usage.

## System Requirements

For Llama 4 Scout with 400K tokens:
- GPU with 80GB+ VRAM (recommended)
- 64GB+ RAM for CPU offloading

For Gemini API testing:
- Internet connection
- Gemini API key

## Documentation

For detailed information:

- [iRoPE Findings](Documentation/irope_findings.md): Technical details on iRoPE implementation and evaluation
- [API Testing Guide](models/gemini/README.md): Guide to using Gemini API for long context testing
- [Memory Optimization](Documentation/memory_optimization.md): Techniques for handling extremely long contexts

## License

This project is licensed under the MIT License - see the LICENSE file for details.
