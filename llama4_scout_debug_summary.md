# Debugging Summary: Llama 4 Scout 10M Context Test on Cloud GPU

**Date:** 2025-04-19

## 1. Project Goal

The objective is to run the `long_context_test.py` script to evaluate the long-context retrieval capabilities (specifically "Needle-in-Haystack" or NIAH) of Meta's Llama 4 Scout model, targeting its claimed 10 million token context window. This involves using a cloud GPU instance capable of handling the model's memory requirements.

## 2. Initial Setup & Requirements

*   **Script:** `long_context_test.py` (from the `llama4-long-context` repository).
*   **Key Dependencies:** `torch` (with CUDA), `transformers`, `accelerate`, `bitsandbytes`.
*   **Critical Feature:** 4-bit quantization (`USE_4BIT_QUANTIZATION = True`) using `bitsandbytes` is essential to reduce the model's significant memory footprint.
*   **Hardware:** High-VRAM GPU required (H100 80GB+ recommended).

## 3. Instances Attempted

1.  **NVIDIA GH200 (ARM64 + H100 96GB VRAM):** Chosen initially for its VRAM and attractive pricing.
2.  **NVIDIA H100 PCIe (x86_64, 80GB VRAM):** Switched to this instance due to compatibility issues on ARM64. This is the currently used instance.

## 4. Errors Encountered & Troubleshooting Steps

A chronological summary of the main roadblocks:

### 4.1. GH200 (ARM64) Instance Issues

*   **Error:** Failed to install `bitsandbytes` with GPU support via pip.
    *   **Troubleshooting:** Confirmed it was commented out in `requirements.txt`, uncommented it.
    *   **Troubleshooting:** Attempted direct pip install (`pip install bitsandbytes --no-cache-dir`). This installed a version.
    *   **Error:** Python import check revealed the installed `bitsandbytes` was CPU-only (`The installed version of bitsandbytes was compiled without GPU support`).
    *   **Troubleshooting:** Attempted compiling `bitsandbytes` from source using CMake and Make for ARM64 + CUDA 12.5. Compilation seemed successful.
    *   **Error:** Python import check *still* showed CPU-only version loaded (likely linking or compilation issue specific to ARM64/CUDA setup).
*   **Error:** Failed to install a CUDA-enabled PyTorch version (`>=2.2`) via pip. Standard CUDA wheels (`cu121`) were older versions, and direct install pulled CPU-only builds.
*   **Decision:** Abandoned ARM64 instance due to fundamental library compatibility issues for CUDA-enabled PyTorch and `bitsandbytes`.

### 4.2. H100 (x86_64) Instance Issues

*   **Setup:** Successfully connected, cloned repo, created venv, installed requirements (including CUDA-enabled PyTorch 2.7.0+cu126 and GPU-enabled `bitsandbytes` 0.45.5 via pip). Verified GPU (`nvidia-smi`) and environment.
*   **Error (Persistent):** Script consistently failed with `OSError: meta-llama/Llama-4-Scout-10M-hf is not a local folder... Repository Not Found`.
    *   **Investigation:** `cat long_context_test.py` showed the *correct* `MODEL_ID` ("...17B-16E-Instruct") was present in the file on disk. However, script execution logs showed it was using the *old* placeholder ID ("...10M-hf").
    *   **Troubleshooting:** Added debug prints (some didn't appear in logs, indicating an old version was running).
    *   **Troubleshooting:** Cleared Python cache (`__pycache__`, `*.pyc`).
    *   **Troubleshooting:** Ran script with absolute path.
    *   **Troubleshooting:** Recreated virtual environment (`.venv`).
    *   **Troubleshooting:** Pushed corrected code from local machine to GitHub, then `git pull` on instance. Script still ran with old ID.
    *   **Resolution:** Used `git checkout HEAD -- long_context_test.py` to forcefully restore the file from the git index. This revealed the *committed* version actually still had the placeholder ID. Required another local fix, push, and `git pull` on the instance.
*   **Error:** `ValueError: The checkpoint ... has model type \`llama4\` but Transformers does not recognize this architecture`.
    *   **Troubleshooting:** Upgraded `transformers` library (`pip install --upgrade transformers`). (Initial attempt ran locally by mistake, then corrected to run on H100).
*   **Error:** `ImportError: No module named 'triton'`.
    *   **Troubleshooting:** Installed `triton` (`pip install triton`).
*   **Error:** `CUDA out of memory` during `model.generate(...)` (inference stage).
    *   **Investigation:** Realized `USE_4BIT_QUANTIZATION` was inadvertently `False` in the script being executed. KV cache for ~197k tokens exceeded VRAM without quantization.
    *   **Troubleshooting:** Set `USE_4BIT_QUANTIZATION = True` in the script.
*   **Error (Current Blocker):** `CUDA out of memory` during `AutoModelForCausalLM.from_pretrained(...)` (model loading stage, specifically `Loading checkpoint shards...`), even with:
    *   `USE_4BIT_QUANTIZATION = True`.
    *   `TARGET_CHAR_COUNT` reduced to `10_000`.
    *   `max_memory={0: "75GiB"}` added to `from_pretrained`.
    *   (Initial attempt at `max_memory` caused `ValueError: size auto is not in a valid format` due to `"cpu": "auto"`, which was corrected).

## 5. Current Status & Hypothesis

*   The environment (Python, venv, CUDA, PyTorch, transformers, accelerate, bitsandbytes, triton) seems correctly set up on the H100 80GB instance.
*   The script `long_context_test.py` is the correct version and attempts to use 4-bit quantization.
*   The script now fails with a CUDA OOM error *during the model loading phase* (`Loading checkpoint shards...`), even with a minimal context target and explicit memory limits.
*   **Hypothesis:** The memory required to simply load the 4-bit quantized Llama 4 Scout model weights using `transformers` with `device_map="auto"` (even with `max_memory`) slightly exceeds the 80GB VRAM available on this H100 instance.

## 6. Possible Next Steps

*   **Confirm Memory Requirements:** Search for reliable benchmarks or reports on the actual VRAM needed to load Llama 4 Scout 17Bx16E with 4-bit quantization using Hugging Face `transformers`. Is 80GB known to be insufficient?
*   **Advanced Loading (`accelerate`):** Explore using a more detailed `accelerate` configuration (via a config file or `infer_auto_device_map`) to potentially offload more layers to CPU RAM during loading, possibly requiring more system RAM.
*   **Different Quantization:** Investigate if other quantization libraries or methods (e.g., AWQ, GPTQ, if compatible models exist) have lower loading overhead.
*   **Larger GPU:** Test on an instance with more VRAM (e.g., > 90GB).
*   **Multi-GPU:** Adapt the setup for a multi-GPU instance (e.g., 2x H100 80GB), allowing `accelerate` to distribute the load.

This debugging journey has been complex due to initial ARM64 compatibility issues, followed by perplexing file state inconsistencies, and now hitting the apparent VRAM limit even with quantization. 