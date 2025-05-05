# iRoPE Testing Guide

This guide explains how to test iRoPE's long context capabilities on your system, specifically targeting the Llama 4 Scout model.

## Setup Steps

1. **Hugging Face Authentication**:
   ```
   ./setup_huggingface.sh
   ```
   This script will guide you through setting up authentication with Hugging Face to access the Llama 4 model.

2. **Environment Preparation**:
   Ensure you have all required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. **Optional Libraries**:
   For optimal performance, you may want to install these libraries:
   - `bitsandbytes`: Enables 4-bit quantization to reduce memory usage
   - `flash-attn`: Optimizes attention computation

## Running Tests

We provide two testing scripts:

### 1. Simple iRoPE Test

For initial testing with a small context (~1250 tokens):

```
./simple_irope_test.py
```

This is useful to verify:
- Model access works
- Basic functionality of iRoPE
- Memory usage patterns

### 2. Full 400K Context Test

For testing the full 400K token capability:

```
python long_context_test.py
```

This script:
- Creates a 1.6M character document (~400K tokens)
- Inserts a "needle" at the 50% position
- Tests if the model can retrieve the needle
- Logs detailed metrics to `irope_400k_test_results.json`

## Troubleshooting

### Authentication Issues

If you encounter a 401 error:
```
❌ Error loading model: 401 Client Error
```

Run the authentication script:
```
./setup_huggingface.sh
```

### Memory Issues

If you encounter CUDA out of memory errors:
1. Reduce the context size in the script (e.g., `TARGET_CHAR_COUNT = 800000`)
2. Disable certain features in the script:
   ```python
   # In long_context_test.py
   USE_4BIT_QUANTIZATION = False
   USE_FLASH_ATTENTION = False
   ```

### Missing Libraries

The scripts are designed to degrade gracefully if optional libraries are missing, but for best results install:
```
pip install bitsandbytes==0.42.0 flash-attn==2.5.5
```

## Results & Next Steps

After successful testing, you can:
1. Analyze the results in `irope_400k_test_results.json`
2. Scale to larger context sizes if your hardware supports it
3. Tune parameters like temperature scaling and RoPE configurations

## Hardware Recommendations

For 400K tokens:
- At least 80GB GPU memory (e.g., A100 or H100)
- 64GB+ RAM if using CPU offloading

For smaller tests (100K tokens):
- 24GB+ GPU memory may be sufficient
- 32GB+ RAM recommended 