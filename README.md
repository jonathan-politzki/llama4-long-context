# Llama 4 Scout - 10M Token Context Window Evaluation

**Version:** 0.1.0

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
├── long_context_test.py    # Main script for generation, inference, and evaluation
├── requirements.txt        # Python dependencies
└── README.md               # This documentation file
```

## 4. Configuration

Key parameters can be adjusted within `long_context_test.py`:

*   `TARGET_CHAR_COUNT` (int): The approximate number of characters to generate for the haystack. Aim for ~4 characters per token (e.g., `40_000_000` for 10M tokens). **Start small (e.g., `10_000`) to test the pipeline before attempting massive contexts.**
*   `NEEDLE` (str): The specific piece of information to hide in the haystack.
*   `QUESTION` (str): The question posed to the LLM, designed to be answerable only by retrieving the `NEEDLE`.
*   `MODEL_ID` (str): **Crucial.** The Hugging Face Hub identifier for the model to test. **Must be updated from the placeholder (`"meta-llama/Llama-4-Scout-10M-hf"`) to the actual released ID.**
*   `USE_4BIT_QUANTIZATION` (bool): Set to `True` to enable 4-bit loading via `bitsandbytes` (requires compatible hardware/OS and the library installed). Set to `False` to load in default precision (typically `bfloat16` as specified in the script), requiring significantly more VRAM.
*   `Generation Parameters` (within `model.generate` call): Parameters like `max_new_tokens`, `do_sample`, `temperature`, etc., can be tuned to influence the model's output generation behavior.

## 5. Setup Instructions

**5.1. Prerequisites:**

*   **Hardware:**
    *   **GPU:** Mandatory for practical execution. An **NVIDIA H100 (80GB VRAM)** or equivalent accelerator is highly recommended, especially for contexts approaching 10M tokens, even with 4-bit quantization. The primary bottleneck is the **KV Cache**, whose memory requirement scales linearly with context length and dominates VRAM usage.
    *   **RAM:** A large amount of system RAM (e.g., 128GB+) is advisable to handle data loading, potential model layer offloading (`device_map="auto"`), and general system overhead.
    *   **Storage:** Sufficient disk space for the virtual environment, libraries, and potentially downloading large model weights.
*   **Software:**
    *   Python (>= 3.8 recommended).
    *   `git` (for cloning).
    *   NVIDIA Drivers, CUDA Toolkit, cuDNN (if using NVIDIA GPU). Ensure compatibility between drivers, CUDA version, and the installed `torch` version.

**5.2. Installation Steps:**

1.  **Clone Repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
2.  **Create & Activate Virtual Environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate # Linux/macOS
    # OR: .venv\Scripts\activate # Windows
    ```
3.  **Install Dependencies:**
    ```bash
    # Upgrade pip (optional but recommended)
    pip install --upgrade pip
    # Install requirements
    pip install -r requirements.txt
    ```
    *   **Note on `torch`:** `requirements.txt` lists `torch` generically. `pip` will attempt to install a compatible version. For GPU acceleration, you might need to install a specific version matching your CUDA toolkit *before* running `pip install -r requirements.txt`. Refer to the [PyTorch website](https://pytorch.org/get-started/locally/) for specific commands.
    *   **Note on `bitsandbytes`:** This library enables 4-bit quantization. It is commented out by default in `requirements.txt` as pre-compiled wheels are often unavailable for macOS or Windows. To enable it:
        *   Uncomment the line in `requirements.txt`.
        *   Ensure you are on a **Linux system with an NVIDIA GPU and compatible CUDA setup**.
        *   Run `pip install -r requirements.txt` again.
        *   Set `USE_4BIT_QUANTIZATION = True` in the script.
4.  **Hugging Face Login:**
    Authenticate to download models:
    ```bash
    huggingface-cli login
    ```
    Enter your access token when prompted.

## 6. Running the Test

1.  **Verify Configuration:** Double-check `MODEL_ID`, `TARGET_CHAR_COUNT`, and `USE_4BIT_QUANTIZATION` in `long_context_test.py`.
2.  **Ensure Environment:** Confirm you are on suitable hardware and the virtual environment is activated.
3.  **Execute Script:**
    ```bash
    python long_context_test.py
    ```
4.  **Monitor Output:**
    *   The script provides verbose output about generation, model loading, tokenization, inference, and evaluation.
    *   **Expect extremely long run times**, potentially hours, for multi-million token contexts, even on H100 hardware.
    *   Watch for CUDA Out-of-Memory (OOM) errors or other exceptions.
5.  **Analyze Results:** Review the final printed model response and the "✅ Success" or "❌ Failure" evaluation message.

## 7. Evaluation Method

The current evaluation is rudimentary:

*   It performs a case-insensitive check for the exact `NEEDLE` string within the generated `response_text`.

This confirms basic retrieval but doesn't assess the *quality* or *contextual appropriateness* of the response beyond finding the needle.

## 8. Troubleshooting

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

## 9. Limitations

*   **Simplistic Haystack:** Uses basic text repetition, which may not accurately reflect the complexity of real-world documents.
*   **Fixed Needle Placement:** Inserts the needle randomly, but doesn't systematically test different positions (e.g., start, middle, end) which can affect performance.
*   **Basic Evaluation:** Only checks for the presence of the needle string.
*   **Hardware Dependency:** Requires high-end, specific GPU hardware, limiting accessibility.
*   **Model Availability:** Relies on the target LLM being released and accessible via Hugging Face Hub.
*   **Static Configuration:** Parameters are hardcoded; adding command-line arguments would increase flexibility.

## 10. Future Work & Potential Improvements

*   **Advanced Haystack Generation:** Incorporate more diverse and realistic text sources (e.g., Wikipedia dumps, ArXiv papers, code repositories).
*   **Systematic Needle Placement:** Add options to place the needle at specific relative positions within the context (e.g., 0%, 25%, 50%, 75%, 100%).
*   **Multiple Needles:** Test the model's ability to retrieve multiple distinct pieces of information.
*   **Sophisticated Evaluation:** Implement metrics beyond simple string matching (e.g., ROUGE scores if the task is summarization-like, exact match on specific extracted answers, semantic similarity checks).
*   **Command-Line Interface:** Use `argparse` to allow configuration via command-line arguments.
*   **API Integration:** Add adapters to run tests against commercial LLM APIs (e.g., Anthropic, OpenAI, Google AI, Groq) that might support long contexts, allowing comparison.
*   **Performance Benchmarking:** Measure and log VRAM usage, inference latency, and token throughput.
*   **Context Length Sweeping:** Automate running tests across a range of `TARGET_CHAR_COUNT` values to plot performance vs. context length.
*   **Visualization:** Generate plots showing success rate vs. context length or needle position.

## 11. Contributing

(Placeholder - detail contribution guidelines if applicable, e.g., pull requests, issue reporting).

## 12. License

(Placeholder - e.g., MIT License, Apache 2.0).
