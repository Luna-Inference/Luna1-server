# RKLLM-Basic

A simple OpenAI-compatible API server for interacting with RKLLM models on Rockchip platforms.

## Quick Start

### Configuration
All settings can be configured in `config.py`, including:
- Model path
- Target platform
- CPU/GPU settings
- Context length and other model parameters

### Running the OpenAI-Compatible Server
```bash
python3 openai_server.py
```
The server uses settings from `config.py` by default. No command-line arguments are required.

## OpenAI API Compatibility

The server implements key OpenAI API endpoints:
- `/v1/chat/completions` - For chat interactions
- `/v1/completions` - For text completions
- `/health` - For server status checks

## Using with OpenAI Client Libraries

You can use standard OpenAI client libraries by configuring them to point to your local server:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-rkllm-api-key"  # Default API key
)

completion = client.chat.completions.create(
    model="rkllm",  # Model name (ignored by server)
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about RKLLM."}  
    ]
)

print(completion.choices[0].message.content)
```

### Client Examples

The `client` directory contains example scripts for interacting with the server:

```bash
# From the rkllm-basic directory:
cd client
python3 completion.py  # Basic completion example
```

## API Endpoints

The server implements the following OpenAI-compatible API endpoints:

### Chat Completions
- **POST /v1/chat/completions**
  - Request body:
  ```json
  {
    "model": "rkllm",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Tell me about RKLLM."}
    ],
    "stream": false
  }
  ```
  - Response format matches OpenAI's API format:
  ```json
  {
    "id": "chatcmpl-123abc",
    "object": "chat.completion",
    "created": 1686000000,
    "model": "rkllm",
    "choices": [{
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "RKLLM is a framework..."
      },
      "finish_reason": "stop"
    }]
  }
  ```

### Text Completions
- **POST /v1/completions**
  - Request body:
  ```json
  {
    "model": "rkllm",
    "prompt": "Tell me about RKLLM."
  }
  ```
  - Response follows OpenAI's API format

### Health Check
- **GET /health**
  - Response: `{"status": "healthy"}`

## Server Configuration

You can pass the following command line arguments when starting the server:

```bash
python3 openai_server.py --rkllm_model_path /path/to/model --target_platform rk3588 --port 8080
```

All parameters have defaults in `config.py` if not specified.
