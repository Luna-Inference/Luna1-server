# RKLLM-Basic

A simple client-server system for interacting with RKLLM models on Rockchip platforms.

## Quick Start

### Configuration
All settings can be configured in `config.py`, including:
- Model path
- Target platform
- CPU/GPU settings
- Context length and other model parameters

### Running the Server
```bash
python3 flask_server.py
```
The server uses settings from `config.py` by default. No command-line arguments are required.

### Using the Client
```bash
python3 flask_client.py
```
Enter your questions at the prompt to chat with the model.

## API Endpoints

The Flask server exposes a single endpoint:

- **POST /rkllm_chat**
  - Request body: `{"messages": [{"role": "user", "content": "your message"}], "stream": true|false}`
  - Response: JSON with complete response or streaming chunks
  
The client automatically formats requests to this endpoint.
