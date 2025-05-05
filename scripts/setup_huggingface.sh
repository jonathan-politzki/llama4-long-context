#!/bin/sh
# Helper script to set up Hugging Face authentication for accessing gated models

echo "===== Hugging Face Authentication Setup ====="
echo "This script will help you authenticate with Hugging Face to access models like Llama 4."
echo

# Check if already logged in
if [ -f ~/.huggingface/token ]; then
  echo "Found existing Hugging Face token."
  echo "Current token: $(cat ~/.huggingface/token | cut -c1-5)...$(cat ~/.huggingface/token | cut -c-5)"
  echo
  printf "Do you want to use this token? (y/n): "
  read use_existing
  if [ "$use_existing" = "y" ] || [ "$use_existing" = "Y" ]; then
    # Export to environment variable as well
    export HF_TOKEN=$(cat ~/.huggingface/token)
    echo "Using existing token. Also exported as HF_TOKEN environment variable."
    exit 0
  fi
fi

echo
echo "Please provide your Hugging Face token."
echo "You can find/create your token at: https://huggingface.co/settings/tokens"
echo

printf "Enter your Hugging Face token: "
read token

# Create directory if it doesn't exist
mkdir -p ~/.huggingface

# Save token to file
echo "$token" > ~/.huggingface/token
echo "Token saved to ~/.huggingface/token"

# Export to environment variable
export HF_TOKEN=$token
echo "Token also exported as HF_TOKEN environment variable for current session."

# Try to use huggingface-cli if available
if command -v huggingface-cli >/dev/null 2>&1; then
  echo "Running huggingface-cli login..."
  echo $token | huggingface-cli login
fi

echo
echo "===== Setup Complete ====="
echo "To use this token in future sessions, add this to your shell config:"
echo "export HF_TOKEN=$token"
echo
echo "Now you should be able to access gated models like Llama 4." 