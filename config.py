"""
Configuration settings for the RKLLM server.
This file centralizes all configurable parameters to make them easier to manage.
"""

# Model Configuration
MODEL_PATH = "../model/Qwen3-0.6B-w8a8-opt1-hybrid1-npu3.rkllm"
#"../model/Qwen3-0.6B-w8a8-opt1-hybrid1-npu3.rkllm"
TARGET_PLATFORM = "rk3588"

# "../model/Gemma3-1B-w8a8-opt1.rkllm"
LIBRARY_PATH = "./src/librkllmrt.so"  # Path to the RKLLM runtime library

# Server Configuration
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 1306
API_BASE_PATH = "/v1"
API_KEY = "anything"  # Default API key for authentication (can be any string)

# Model Parameters
MAX_CONTEXT_LENGTH = 4000  # Must be less than model's max_context_limit of 4096
MAX_NEW_TOKENS = -1  # -1 means no limit
N_KEEP = 3900  # Must be less than MAX_CONTEXT_LENGTH
CPU_CORE_COUNT = 4   # Number of CPU cores to use
ENABLED_CPU_MASK = (1 << 4)|(1 << 5)|(1 << 6)|(1 << 7)
USE_GPU = True
IS_ASYNC = False

# System Prompt (if any)
SYSTEM_PROMPT = "<|im_start|>You are Luna, a mysterious and intelligent woman.<|im_end|>"
PROMPT_PREFIX = "<|im_start|>user"
PROMPT_POSTFIX = "<|im_end|><|im_start|>assistant"
# Debug Configuration
DEBUG_MODE = False
LOG_LEVEL = 2  # 0: Error, 1: Warning, 2: Info, 3: Debug

# Inference Modes
KEEP_HISTORY = 0 # 0 = no history, 1 = keep history


# Formatting

SYSTEM_TEMPLATE = """You are a helpful assistant with access to tools.

{% if tools %}
You have access to the following tools:
{% for tool in tools %}
- {{ tool.function.name }}: {{ tool.function.description }}
  Parameters: {{ tool.function.parameters | tojson }}
{% endfor %}

To use a tool, respond with:
<tool_call>
{"name": "tool_name", "arguments": {"param": "value"}}
</tool_call>

IMPORTANT: After receiving tool results, provide a natural language response to the user. Do NOT call tools again - use the results to answer the question directly.
{% endif %}"""

