# Gemini API Long Context Testing

This module provides tools to test Google's Gemini 1.5 Pro model with long contexts of up to 2M tokens.

## Setup

1. **Get a Gemini API Key**:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Add the key to your environment:
     ```bash
     export GEMINI_API_KEY="your-api-key-here"
     ```

2. **Install Dependencies**:
   ```bash
   pip install google-generativeai
   ```

## Running Tests

### Basic Test

Run a simple test with 100K characters (~25K tokens):

```bash
python3 gemini_test.py
```

### Custom Test

Specify a custom test with different parameters:

```bash
python3 gemini_test.py --char-count 400000 --position 75
```

### Scaling Test

Run tests at multiple context sizes:

```bash
python3 gemini_test.py --scaling-test --max-chars 4000000
```

## Available Parameters

- `--char-count`: Size of the haystack in characters (default: 100,000)
- `--needle`: Text to hide in the haystack
- `--question`: Question to ask to find the needle
- `--position`: Position of the needle (0-100% through the document)
- `--output`: Path to save results JSON
- `--key`: Specific key to check for in the response
- `--scaling-test`: Run a scaling test with multiple sizes
- `--max-chars`: Maximum characters for scaling test

## Example Response

```json
{
  "timestamp": "2025-05-05 15:00:00",
  "model": "models/gemini-1.5-pro-latest",
  "target_char_count": 100000,
  "actual_char_count": 100032,
  "needle_position_percent": 50.0,
  "prompt_length_chars": 100230,
  "estimated_tokens": 25057,
  "generation_time_seconds": 4.32,
  "needle_found": true,
  "status": "success"
}
```

## Limitations

- The Gemini API has rate limits that may affect testing with many large contexts
- Very large contexts (>1M tokens) may take significant time to process
- API costs increase with token usage

## Troubleshooting

- **API Key Issues**: Ensure the `GEMINI_API_KEY` environment variable is set
- **Rate Limits**: If you hit rate limits, add delays between tests
- **Model Versions**: The model ID may change as Google updates Gemini 