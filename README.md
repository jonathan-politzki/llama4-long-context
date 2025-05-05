# Llama 4 Scout - 10M Token Context Window Evaluation

**Version:** 0.3.0

## 1. Project Goal & Significance

This project provides a framework for evaluating the long-context retrieval capabilities of large language models (LLMs), with a specific focus on Meta's Llama 4 Scout model and its claimed 10 million token context window.

The ability of LLMs to process and recall information from extremely long inputs is crucial for applications like:

*   Analyzing entire codebases.
*   Summarizing and querying large document sets (e.g., research papers, legal documents, books).
*   Maintaining coherent, long-running conversations.
*   Processing extensive user histories for personalization.

This project utilizes the **"Needle-in-Haystack" (NIAH)** evaluation methodology. This standard technique assesses an LLM's ability to locate and retrieve a specific, small piece of information (the "needle") deliberately hidden within a large, potentially distracting corpus of text (the "haystack"). Success in this test is a strong indicator of the model's practical usability for long-context tasks.

## 2. Workflow & Architecture

The core logic resides in the `long_context_test.py` script, which performs the following steps:

1.  **Configuration Loading:** Reads parameters like target context size, model ID, needle text, and question.
2.  **Haystack Generation:** Creates a large block of filler text (`generate_filler_text`) based on the `TARGET_CHAR_COUNT`. *Currently uses simple text repetition.*
3.  **Needle Injection:** Randomly inserts the predefined `NEEDLE` string into the haystack (`insert_needle`), demarcated by markers (`--- NEEDLE START ---`, `--- NEEDLE END ---`) for clarity during generation (these markers are part of the context fed to the LLM).
4.  **Prompt Formatting:** Constructs the final prompt (`create_prompt`) by wrapping the combined haystack-and-needle text within `<document>` tags and appending the specific `QUESTION`.
5.  **Resource Preparation:** Attempts to clear system memory before loading the large model.
6.  **Model & Tokenizer Loading:** Initializes the Hugging Face `transformers` tokenizer and model specified by `MODEL_ID`. Critically, this step:
    *   Uses `AutoTokenizer` and `AutoModelForCausalLM`.
    *   Supports optional 4-bit quantization via `bitsandbytes` (`USE_4BIT_QUANTIZATION=True`) to reduce memory footprint, crucial for fitting Llama 4 Scout on target hardware.
    *   Utilizes `device_map="auto"` (via the `accelerate` library) to distribute the model across available hardware (GPUs, CPU, RAM) automatically.
    *   Requires `trust_remote_code=True` as is common for complex model architectures.
7.  **Tokenization:** Converts the large prompt string into input tokens understandable by the model.
8.  **Inference:** Runs the `model.generate()` function to produce an answer based on the provided context. This is the most computationally and memory-intensive step, especially with multi-million token contexts.
9.  **Decoding:** Converts the model's output tokens back into human-readable text.
10. **Evaluation:** Performs a simple case-insensitive substring check to see if the `NEEDLE` text is present in the model's generated response.
11. **Output & Cleanup:** Prints the model's response and the evaluation result. Attempts to release model and tensor resources from memory.

## 3. File Structure

```
.
├── .venv/                  # Python virtual environment (if created locally)
├── long_context_test.py    # Main script for Llama 4 Scout evaluation
├── model_comparison.py     # Script to compare Llama 4 vs Gemini
├── setup_and_run.sh        # Helper script to set up and run the comparison
├── Dockerfile              # Docker container definition
├── docker-compose.yml      # Docker Compose configuration
├── run-docker.sh           # Helper script to run Docker commands
├── requirements.txt        # Python dependencies
├── offload_folder/         # Created at runtime for model offloading
├── comparison_results/     # Created at runtime to store comparison results
├── hf_cache/               # HuggingFace cache directory
└── README.md               # This documentation file
```

## 4. Dockerized Deployment

### 4.1 Using Docker (Recommended)

The project is dockerized for easy deployment across different environments and GPU instances.

**Prerequisites:**
- Docker and Docker Compose installed
- NVIDIA Container Toolkit (for GPU support)
- HuggingFace account and access token (for downloading Llama 4 Scout model)
- Gemini API key (optional, for comparison)

**Running with Docker:**

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd llama4-long-context
   ```

2. **Make the run script executable:**
   ```bash
   chmod +x run-docker.sh
   ```

3. **Run the tests using the helper script:**
   ```bash
   # Show help
   ./run-docker.sh
   
   # Run interactive setup
   ./run-docker.sh setup
   
   # Run comparison with specific context size (in test mode)
   ./run-docker.sh compare 100000
   
   # Run full comparison (no test mode)
   ./run-docker.sh compare-full 1000000
   
   # Run Llama test only
   ./run-docker.sh llama
   
   # Get a shell in the container
   ./run-docker.sh shell
   ```

4. **Set Gemini API key for comparison (optional):**
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ./run-docker.sh setup
   ```

5. **Results will be saved to:**
   ```
   ./comparison_results/
   ```

### 4.2 Manual Setup (Alternative)

Alternatively, you can run the scripts directly:

1.  **Clone Repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
2.  **Use the Automated Setup Script:**
    ```bash
    ./setup_and_run.sh
    ```
    This script will:
    - Create a Python virtual environment
    - Install all required dependencies
    - Run the model comparison with your chosen settings
    
    Alternatively, follow the manual setup:

3.  **Create & Activate Virtual Environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate # Linux/macOS
    # OR: .venv\Scripts\activate # Windows
    ```
4.  **Install Dependencies:**
    ```bash
    # Upgrade pip (optional but recommended)
    pip install --upgrade pip
    # Install requirements
    pip install -r requirements.txt
    ```
    
5.  **Hugging Face Login:**
    Authenticate to download models:
    ```bash
    huggingface-cli login
    ```
    Enter your access token when prompted.

## 5. Configuration

Key parameters can be adjusted within `long_context_test.py`:

*   `TARGET_CHAR_COUNT` (int): The approximate number of characters to generate for the haystack. Aim for ~4 characters per token (e.g., `40_000_000` for 10M tokens). **Start small (e.g., `10_000`) to test the pipeline before attempting massive contexts.**
*   `NEEDLE` (str): The specific piece of information to hide in the haystack.
*   `QUESTION` (str): The question posed to the LLM, designed to be answerable only by retrieving the `NEEDLE`.
*   `MODEL_ID` (str): The Hugging Face Hub identifier for the model.
*   `USE_4BIT_QUANTIZATION` (bool): Set to `True` to enable 4-bit loading via `bitsandbytes` (requires compatible hardware/OS and the library installed).
*   `ENABLE_CPU_OFFLOAD` (bool): Enable offloading model layers to CPU when needed.
*   `MAX_GPU_MEMORY` (str): Limit GPU memory usage to this amount.
*   `Generation Parameters` (within `model.generate` call): Parameters like `max_new_tokens`, `do_sample`, `temperature`, etc., can be tuned to influence the model's output generation behavior.

## 6. Running the Tests

### 6.1 Llama 4 Scout Long Context Test

To run the original Llama 4 Scout 10M context test:

```bash
# With Docker:
./run-docker.sh llama

# Without Docker:
python long_context_test.py
```

### 6.2 Llama 4 vs Gemini Comparison

The comparison script allows you to benchmark Llama 4 Scout against Google's Gemini model:

```bash
# With Docker:
export GEMINI_API_KEY="your-api-key-here"
./run-docker.sh compare 100000  # Test mode with 100K chars
./run-docker.sh compare-full 100000  # Full inference with 100K chars

# Without Docker:
# Set your Gemini API key
export GEMINI_API_KEY="your-api-key-here"

# Run the comparison with default settings
python model_comparison.py

# Or specify custom parameters
python model_comparison.py --char-count 100000 --llama-only --test-mode
python model_comparison.py --char-count 1000000 --gemini-only --gemini-api-key "your-key"
```

For convenience, you can use the automated script:

```bash
./setup_and_run.sh
```

This will guide you through the setup and running process interactively.

## 7. Scaling Strategy

To tackle the 10M token context challenge, we recommend this progressive approach:

1. **Start Small**: Begin with 10,000 characters (~2.5K tokens) to test basic functionality
2. **Gradual Scaling**: Incrementally increase to 100K, then 1M, then 8M characters
3. **Test Mode First**: Use `--test-mode` to verify model loading and processing without full inference
4. **Hardware Scaling**: 
   - For 2M tokens: Single H100 GPU (80GB) with heavy CPU offloading
   - For 10M tokens: Multiple GPUs (2-4× H100s) or specialized hardware

Each successful test validates the model's functionality at that scale before moving to larger contexts.

## 8. Memory Optimization

To address memory constraints, especially when running on cloud GPUs like H100, the scripts implement several optimizations:

- **4-bit Quantization**: Reduces model weight memory by loading in 4-bit precision using bitsandbytes
- **CPU Offloading**: Automatically offloads less frequently used model layers to system RAM
- **Memory Limits**: Sets explicit memory limits to avoid OOM errors
- **Chunked Processing**: Processes very long contexts in smaller manageable chunks
- **Selective Windows**: For generation, uses a context window around the needle location
- **Resource Monitoring**: Tracks system and GPU resources throughout execution
- **Memory Management**: Properly frees resources after use to prevent memory leaks

For particularly large contexts (approaching 10M tokens), you may need:
- A multi-GPU setup with 2 or more H100 GPUs
- A system with 128GB+ of RAM for CPU offloading
- Appropriate CUDA configuration to prevent memory fragmentation

## 9. Model Comparison

The `model_comparison.py` script enables side-by-side evaluation of:

1. **Llama 4 Scout**: Using local inference with 4-bit quantization for 2M token contexts
2. **Gemini 1.5 Pro**: Using Google's API for 2M token contexts

The script records:
- Success/failure in retrieving the needle
- Runtime performance
- Memory usage
- Response quality

Results are saved to JSON files in the `comparison_results/` directory for later analysis.

## 10. Hardware Requirements

**For 10M token contexts with Llama 4 Scout:**
- **GPU Memory**: ~320-400GB (combined) - due to KV cache scaling linearly with context length
- **System RAM**: 128GB+ for CPU offloading and model operations
- **Storage**: 20GB+ for model weights and cache
- **Recommended Setup**: 4× H100 GPUs (80GB each) or equivalent

**For 2M token contexts with Llama 4 Scout:**
- **GPU Memory**: 80GB minimum with heavy CPU offloading
- **System RAM**: 64GB+ for CPU offloading
- **Recommended Setup**: 1× H100 GPU (80GB) or equivalent

These requirements are based on Meta's research on long-context models and practical testing.

## 11. Troubleshooting

*   **`CUDA out of memory`:**
    *   The most common issue with large contexts.
    *   **Solutions:** Reduce `TARGET_CHAR_COUNT`, ensure 4-bit quantization is enabled (`USE_4BIT_QUANTIZATION = True`) and `bitsandbytes` is correctly installed/functional, use hardware with more VRAM (H100/equivalent), ensure no other processes are consuming VRAM.
*   **`bitsandbytes` Installation/Runtime Errors:**
    *   Usually due to OS/hardware incompatibility (requires Linux/NVIDIA GPU).
    *   **Solution:** Disable quantization (`USE_4BIT_QUANTIZATION = False`) and remove/comment out `bitsandbytes` from `requirements.txt`. This will significantly increase VRAM requirements.
*   **Model Loading Errors (`OSError`, `HTTPError`):**
    *   Check `MODEL_ID` is correct and accessible on Hugging Face.
    *   Verify Hugging Face login (`huggingface-cli login`).
    *   Check network connectivity.
*   **Very Long Runtimes:**
    *   Expected for large contexts. Inference time scales significantly with context length.
    *   Monitor system resources (`nvidia-smi`, `htop`) to ensure the process is active.
    *   Consider using smaller `TARGET_CHAR_COUNT` for initial tests.
*   **Dependency Conflicts:**
    *   Ensure a clean virtual environment.
    *   Check compatibility between `torch`, `transformers`, `accelerate`, `bitsandbytes`, and your CUDA version.
*   **Gemini API Errors:**
    *   Verify API key is correct
    *   Check API rate limits and quotas
    *   For very large contexts, consider increasing request timeouts
*   **Docker Issues:**
    *   Ensure nvidia-docker is installed and configured correctly
    *   Check `docker info | grep -i runtime` shows nvidia
    *   If CUDA version mismatch occurs, adjust the base image in Dockerfile

## 12. Limitations

*   **Simplistic Haystack:** Uses basic text repetition, which may not accurately reflect the complexity of real-world documents.
*   **Fixed Needle Placement:** Inserts the needle randomly, but doesn't systematically test different positions (e.g., start, middle, end) which can affect performance.
*   **Basic Evaluation:** Only checks for the presence of the needle string.
*   **Hardware Dependency:** Requires high-end, specific GPU hardware, limiting accessibility.
*   **Model Availability:** Relies on the target LLM being released and accessible via Hugging Face Hub.

## 13. Future Work & Potential Improvements

*   **Advanced Haystack Generation:** Incorporate more diverse and realistic text sources (e.g., Wikipedia dumps, ArXiv papers, code repositories).
*   **Systematic Needle Placement:** Add options to place the needle at specific relative positions within the context (e.g., 0%, 25%, 50%, 75%, 100%).
*   **Multiple Needles:** Test the model's ability to retrieve multiple distinct pieces of information.
*   **Sophisticated Evaluation:** Implement metrics beyond simple string matching (e.g., ROUGE scores if the task is summarization-like, exact match on specific extracted answers, semantic similarity checks).
*   **API Integration:** Add adapters to run tests against more commercial LLM APIs that might support long contexts, allowing comparison.
*   **Performance Benchmarking:** Measure and log VRAM usage, inference latency, and token throughput.
*   **Context Length Sweeping:** Automate running tests across a range of `TARGET_CHAR_COUNT` values to plot performance vs. context length.
*   **Visualization:** Generate plots showing success rate vs. context length or needle position.

## 14. Contributing

(Placeholder - detail contribution guidelines if applicable, e.g., pull requests, issue reporting).

## 15. License

(Placeholder - e.g., MIT License, Apache 2.0).
