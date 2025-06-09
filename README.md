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
- `/v1/chat/completions` - For chat interactions with support for tool calling
- `/v1/completions` - For text completions
- `/v1/models` - List available models
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
    model="rkllm-local",  # Default model name
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

### Models
- **GET /v1/models**
  - Lists available models:
  ```json
  {
    "object": "list",
    "data": [
      {
        "id": "rkllm-local",
        "object": "model",
        "created": 1677649963,
        "owned_by": "rkllm"
      }
    ]
  }
  ```

### Tool Calling

The server supports OpenAI-style tool calling. Here's an example of how to use it:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-rkllm-api-key"
)

completion = client.chat.completions.create(
    model="rkllm-local",
    messages=[
        {"role": "user", "content": "What does John Smith do for a living?"}
    ],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_profession",
                "description": "Get the profession of a person",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name of the person"
                        }
                    },
                    "required": ["name"]
                }
            }
        }
    ]
)

print(completion.choices[0].message)
```

#### Built-in Tools

1. **get_profession(name: str)**
   - Returns the profession of a person given their name.

2. **get_current_time_string()**
   - Returns the current time in the format 'H:MMam/pm'.

### Health Check
- **GET /health**
  - Response: `{"status": "healthy"}`

## Server Configuration

You can pass the following command line arguments when starting the server:

```bash
python3 openai_server.py --rkllm_model_path /path/to/model --target_platform rk3588 --port 8080
```

### Additional Configuration

You can also configure the following parameters in `config.py` or via environment variables:

- `LOG_LEVEL`: Logging level (default: 1)
- `MAX_CONTEXT_LENGTH`: Maximum context length for the model
- `MAX_NEW_TOKENS`: Maximum number of new tokens to generate
- `CPU_CORE_COUNT`: Number of CPU cores to use
- `ENABLED_CPU_MASK`: CPU core mask for binding
- `KEEP_HISTORY`: Whether to keep conversation history (0 or 1)

All parameters have defaults in `config.py` if not specified.
