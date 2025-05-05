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

# Install pip dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY long_context_test.py model_comparison.py setup_and_run.sh README.md ./

# Make scripts executable
RUN chmod +x setup_and_run.sh

# Create directories needed by the application
RUN mkdir -p /app/offload_folder /app/comparison_results

# Set environment variable for HF cache
ENV HF_HOME=/app/.cache

# Create a wrapper script to execute with different commands
RUN echo '#!/bin/bash\n\
if [ "$1" = "setup" ]; then\n\
    ./setup_and_run.sh\n\
elif [ "$1" = "llama" ]; then\n\
    python long_context_test.py\n\
elif [ "$1" = "compare" ]; then\n\
    python model_comparison.py --char-count ${2:-10000} --llama-only ${3:---test-mode}\n\
else\n\
    exec "$@"\n\
fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["setup"] 