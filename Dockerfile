FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/conda/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

# Create working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install pip dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install flash-attention for improved performance
RUN pip install --no-cache-dir flash-attn

# Copy project files
COPY *.py *.sh README.md ./

# Make scripts executable
RUN chmod +x *.sh

# Create directories needed by the application
RUN mkdir -p /app/offload_folder /app/comparison_results

# Set environment variable for HF cache
ENV HF_HOME=/app/.cache

# Create entrypoint script
RUN echo '#!/bin/bash\n\
# Login to HuggingFace if token is provided\n\
if [ -n "$HF_TOKEN" ]; then\n\
  echo "Logging in to HuggingFace..."\n\
  huggingface-cli login --token $HF_TOKEN --add-to-git-credential\n\
fi\n\
\n\
# Run command\n\
if [ "$1" = "setup" ]; then\n\
  ./setup_and_run.sh\n\
elif [ "$1" = "llama" ]; then\n\
  python long_context_test.py\n\
elif [ "$1" = "compare" ]; then\n\
  python model_comparison.py --char-count ${2:-10000} --llama-only ${3:-""}\n\
elif [ "$1" = "test" ]; then\n\
  python model_comparison.py --char-count ${2:-10000} --llama-only --test-mode\n\
elif [ "$1" = "flash-test" ]; then\n\
  echo "=== Running Progressive Context Tests with Flash Attention 2 ==="\n\
  \n\
  # Test small context first\n\
  echo "Testing with 100,000 characters (~25,000 tokens)..."\n\
  python model_comparison.py --char-count 100000 --llama-only --test-mode\n\
  if [ $? -ne 0 ]; then exit 1;\n\
  fi\n\
  \n\
  # Test medium context\n\
  echo "Testing with 500,000 characters (~125,000 tokens)..."\n\
  python model_comparison.py --char-count 500000 --llama-only --test-mode\n\
  if [ $? -ne 0 ]; then exit 1;\n\
  fi\n\
  \n\
  # Test large context\n\
  echo "Testing with 1,000,000 characters (~250,000 tokens)..."\n\
  python model_comparison.py --char-count 1000000 --llama-only --test-mode\n\
  if [ $? -ne 0 ]; then exit 1;\n\
  fi\n\
  \n\
  # Test very large context\n\
  echo "Testing with 2,000,000 characters (~500,000 tokens)..."\n\
  python model_comparison.py --char-count 2000000 --llama-only --test-mode\n\
  if [ $? -ne 0 ]; then exit 1;\n\
  fi\n\
  \n\
  # Test extremely large context\n\
  echo "Testing with 4,000,000 characters (~1,000,000 tokens)..."\n\
  python model_comparison.py --char-count 4000000 --llama-only --test-mode\n\
  if [ $? -ne 0 ]; then exit 1;\n\
  fi\n\
  \n\
  # Test with inference for smaller sizes\n\
  echo "Testing inference with 100,000 characters..."\n\
  python model_comparison.py --char-count 100000 --llama-only\n\
  if [ $? -ne 0 ]; then exit 1;\n\
  fi\n\
  \n\
  echo "Testing inference with 500,000 characters..."\n\
  python model_comparison.py --char-count 500000 --llama-only\n\
  if [ $? -ne 0 ]; then exit 1;\n\
  fi\n\
  \n\
  echo "Testing inference with 1,000,000 characters..."\n\
  python model_comparison.py --char-count 1000000 --llama-only\n\
  if [ $? -ne 0 ]; then exit 1;\n\
  fi\n\
  \n\
  echo "All tests completed successfully!"\n\
else\n\
  exec "$@"\n\
fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["setup"] 