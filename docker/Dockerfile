FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 9.0+PTX"
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    # Try to install flash-attention, but don't fail if it doesn't work
    pip install --no-cache-dir flash-attn==2.5.8 || echo "Failed to install flash-attention, continuing without it"

# Copy the application code
COPY . /app/

# Make scripts executable
RUN chmod +x /app/scripts/*.sh
RUN find /app -name "*.py" -type f -exec chmod +x {} \;

# Create directory for results
RUN mkdir -p /app/results

# Default command
CMD ["/bin/bash"] 