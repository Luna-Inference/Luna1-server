# API Endpoint Documentation for server.py

This document provides details about the API endpoints available in `server.py`.

## 1. Chat Completions

-   **Endpoint**: `/v1/chat/completions`
-   **Method**: `POST`
-   **Description**: Provides chat-based completions, similar to OpenAI's chat completion endpoint. It supports streaming responses and tool calls.
-   **Request Body (JSON)**:
    ```json
    {
        "model": "luna-small", // Or other configured model name
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
            // ... more messages
        ],
        "stream": false, // Optional, boolean for streaming
        "tools": [], // Optional, list of tool definitions
        "tool_choice": "auto", // Optional, how to use tools
        // Other OpenAI compatible parameters like temperature, top_p, max_tokens etc.
    }
    ```
-   **Response (JSON, non-streaming)**:
    ```json
    {
        "id": "chatcmpl-...",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "luna-small",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "\n\nHello there, how may I assist you today?"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21
        }
    }
    ```
-   **Response (Server-Sent Events, streaming)**:
    A series of `data:` events, with the final event having `[DONE]` or a `finish_reason`.

## 2. Completions

-   **Endpoint**: `/v1/completions`
-   **Method**: `POST`
-   **Description**: Provides text completions, similar to OpenAI's legacy completion endpoint.
-   **Request Body (JSON)**:
    ```json
    {
        "model": "luna-small",
        "prompt": "Translate the following English text to French: 'Hello world.'",
        "stream": false, // Optional
        // Other OpenAI compatible parameters
    }
    ```
-   **Response (JSON, non-streaming)**:
    ```json
    {
        "id": "cmpl-...",
        "object": "text_completion",
        "created": 1677652288,
        "model": "luna-small",
        "choices": [{
            "index": 0,
            "text": "\n\nBonjour le monde.",
            "logprobs": null,
            "finish_reason": "length"
        }],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 7,
            "total_tokens": 12
        }
    }
    ```
-   **Response (Server-Sent Events, streaming)**:
    Similar to chat completions streaming.

## 3. V1 Compatibility Endpoint

-   **Endpoint**: `/v1/`
-   **Method**: `POST`
-   **Description**: A compatibility route. Based on the code, it appears to redirect or act as an alias for the `/v1/chat/completions` endpoint.
-   **Request/Response**: Assumed to be the same as `/v1/chat/completions`.

## 4. Models

-   **Endpoint**: `/v1/models`
-   **Method**: `GET`
-   **Description**: Returns a list of available models, mimicking OpenAI's API format.
-   **Request Body**: None
-   **Response (JSON)**:
    ```json
    {
        "object": "list",
        "data": [
            {
                "id": "luna-small", // Or other configured model name
                "object": "model",
                "created": 1677628800, // Example timestamp
                "owned_by": "rkllm"
            }
            // Potentially other models
        ]
    }
    ```

## 5. Health Check (Performance Metrics)

-   **Endpoint**: `/health`
-   **Method**: `GET`
-   **Description**: Returns server status **and last-run performance statistics** provided by the RKLLM runtime.
-   **Request Body**: None
-   **Response (JSON)**:
    ```json
    {
        "status": "healthy",
        "generation_status": "idle",          // or "generating"
        "tools_loaded": ["..."],             // dynamically loaded functions
        "prefill_speed_tps": "405.85",      // Tokens-per-second during prompt prefill
        "generation_speed_tps": "27.07",    // Tokens-per-second during answer generation
        "memory_usage_mb": "1524.00"        // Peak RAM usage (MB)
    }
    ```

## 6. Device Recognition

- **Endpoint**: `/luna`
- **Method**: `GET`
- **Description**: Used for device recognition among all devices on the network. Returns a simple JSON object indicating the device type.
- **Request Body**: None
- **Response (JSON)**:
    ```json
    {
        "device": "luna"
    }
    ```

## 7. WiFi Connect

- **Endpoint**: `/wifi`
- **Method**: `POST`
- **Description**: Connect the device to a WiFi network using `nmcli`.
- **Request Body (JSON)**:
    ```json
    {
        "uuid": "Your_WiFi_Name",
        "password": "Your_Password"
    }
    ```
- **Response (JSON)**:
    ```json
    {
        "success": true,
        "stdout": "...",  // Standard output from nmcli
        "stderr": ""        // Standard error from nmcli, if any
    }
    ```

