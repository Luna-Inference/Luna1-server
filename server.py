import ctypes
import sys
import os
import subprocess
import resource
import threading
import time
import argparse
import json
import re
import uuid
import importlib.util
from datetime import datetime

# -------- Token counting helper --------
def count_tokens(text: str) -> int:
    """
    Return the number of tokens in `text` using tiktoken's GPT-2 encoding if
    available. Falls back to a simple whitespace split when tiktoken isn't
    installed so the server continues to run.
    """
    try:
        import tiktoken  # type: ignore
        try:
            encoding = tiktoken.get_encoding("qwen")
            print('qwen encoding found')
        except Exception:
            encoding = tiktoken.get_encoding("gpt2")
            print('gpt2 encoding found')
        return len(encoding.encode(text))
    except Exception:
        # Fallback keeps the server functional even without tiktoken
        return len(text.split())
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from jinja2 import Template
from config import *

app = Flask(__name__)
# Enable CORS for all routes
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS", "PUT", "DELETE", "PATCH"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "supports_credentials": True,
        "max_age": 3600
    }
})

# Set the dynamic library path
rkllm_lib = ctypes.CDLL(LIBRARY_PATH)

# Define the structures from the library
RKLLM_Handle_t = ctypes.c_void_p
userdata = ctypes.c_void_p(None)

LLMCallState = ctypes.c_int
LLMCallState.RKLLM_RUN_NORMAL  = 0
LLMCallState.RKLLM_RUN_WAITING  = 1
LLMCallState.RKLLM_RUN_FINISH  = 2
LLMCallState.RKLLM_RUN_ERROR   = 3

RKLLMInputMode = ctypes.c_int
RKLLMInputMode.RKLLM_INPUT_PROMPT      = 0
RKLLMInputMode.RKLLM_INPUT_TOKEN       = 1
RKLLMInputMode.RKLLM_INPUT_EMBED       = 2
RKLLMInputMode.RKLLM_INPUT_MULTIMODAL  = 3

RKLLMInferMode = ctypes.c_int
RKLLMInferMode.RKLLM_INFER_GENERATE = 0
RKLLMInferMode.RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1
RKLLMInferMode.RKLLM_INFER_GET_LOGITS = 2

os.environ['RKLLM_LOG_LEVEL'] = str(LOG_LEVEL)

# [Keep all the ctypes structure definitions from the original code]
class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("reserved", ctypes.c_uint8 * 106)
    ]

class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam),
    ]

class RKLLMLoraAdapter(ctypes.Structure):
    _fields_ = [
        ("lora_adapter_path", ctypes.c_char_p),
        ("lora_adapter_name", ctypes.c_char_p),
        ("scale", ctypes.c_float)
    ]

class RKLLMEmbedInput(ctypes.Structure):
    _fields_ = [
        ("embed", ctypes.POINTER(ctypes.c_float)),
        ("n_tokens", ctypes.c_size_t)
    ]

class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [
        ("input_ids", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_size_t)
    ]

class RKLLMMultiModelInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t),
        ("n_image", ctypes.c_size_t),
        ("image_width", ctypes.c_size_t),
        ("image_height", ctypes.c_size_t)
    ]

class RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModelInput)
    ]

class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("input_mode", ctypes.c_int),
        ("input_data", RKLLMInputUnion)
    ]

class RKLLMLoraParam(ctypes.Structure):
    _fields_ = [
        ("lora_adapter_name", ctypes.c_char_p)
    ]

class RKLLMPromptCacheParam(ctypes.Structure):
    _fields_ = [
        ("save_prompt_cache", ctypes.c_int),
        ("prompt_cache_path", ctypes.c_char_p)
    ]

class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", RKLLMInferMode),
        ("lora_params", ctypes.POINTER(RKLLMLoraParam)),
        ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam)),
        ("keep_history", ctypes.c_int)
    ]

class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int)
    ]

class RKLLMResultLogits(ctypes.Structure):
    _fields_ = [
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("vocab_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int)
    ]

class RKLLMPerfStat(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("prefill_time_ms", ctypes.c_float),
        ("prefill_tokens", ctypes.c_int),
        ("generate_time_ms", ctypes.c_float),
        ("generate_tokens", ctypes.c_int),
        ("memory_usage_mb", ctypes.c_float),
    ]

class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
        ("logits", RKLLMResultLogits),
        ("perf", RKLLMPerfStat)
    ]

# OpenAI API Configuration
DEFAULT_MODEL_NAME = "luna-small"

# Tool calling configuration
TOOL_REGISTRY = {}  # Will store dynamically loaded tools

# Jinja2 templates for tool formatting
TOOL_SYSTEM_TEMPLATE = Template(SYSTEM_TEMPLATE)

# Create a lock to control multi-user access to the server.
lock = threading.Lock()

# Create a global variable to indicate whether the server is currently in a blocked state.
is_blocking = False

# Thread-safe queue for streaming output
from queue import Queue

# Define global variables to store the callback function output
class GlobalState:
    def __init__(self):
        self.text_queue = Queue()
        self.finished = True  # Start as idle
        self.lock = threading.Lock()
        self.reset_perf_metrics()

    def reset_perf_metrics(self):
        self.prompt_eval_start_time = 0
        self.first_token_time = 0
        self.generation_finish_time = 0
        self.prompt_word_count = 0
        self.generated_word_count = 0
        self.prompt_eval_speed_wps = 0.0
        self.generation_speed_wps = 0.0
        self.prefill_tps = 0.0
        self.generation_tps = 0.0
        self.memory_usage_mb = 0.0

global_state = GlobalState()
split_byte_data = bytes(b"")

def openai_error_response(message, error_type="invalid_request_error", param=None, code=None, status_code=400):
    """Generate OpenAI-compatible error response"""
    return jsonify({
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code
        }
    }), status_code

# Define the callback function
def callback_impl(result, userdata, state):
    global global_state, split_byte_data
    with global_state.lock:
        current_time = time.time()

        if state == 0:  # Normal text output (RKLLM_RUN_NORMAL)
            if global_state.first_token_time == 0:
                global_state.first_token_time = current_time
                eval_duration = global_state.first_token_time - global_state.prompt_eval_start_time
                if eval_duration > 0.001: # Avoid division by zero
                    global_state.prompt_eval_speed_wps = global_state.prompt_word_count / eval_duration
                else:
                    global_state.prompt_eval_speed_wps = float('inf')

            if result.contents.text:
                text_chunk = result.contents.text.decode('utf-8')
                # Count tokens in the generated chunk
                global_state.generated_word_count += count_tokens(text_chunk)
                global_state.text_queue.put(text_chunk)
                print(text_chunk, end="", flush=True)

        elif state == 2:  # Generation finished (RKLLM_RUN_FINISH)
            global_state.generation_finish_time = current_time
            # Ensure first_token_time is set, even for empty responses
            if global_state.first_token_time == 0:
                global_state.first_token_time = current_time

            gen_duration = global_state.generation_finish_time - global_state.first_token_time
            if gen_duration > 0.001 and global_state.generated_word_count > 0:
                global_state.generation_speed_wps = global_state.generated_word_count / gen_duration
            elif global_state.generated_word_count > 0:
                global_state.generation_speed_wps = float('inf')
            else:
                global_state.generation_speed_wps = 0.0
            # Capture RKLLM perf stats
            if result and hasattr(result.contents, 'perf'):
                perf = result.contents.perf
                if perf.prefill_time_ms > 0:
                    global_state.prefill_tps = perf.prefill_tokens / (perf.prefill_time_ms / 1000.0)
                else:
                    global_state.prefill_tps = 0.0
                if perf.generate_time_ms > 0:
                    global_state.generation_tps = perf.generate_tokens / (perf.generate_time_ms / 1000.0)
                else:
                    global_state.generation_tps = 0.0
                global_state.memory_usage_mb = perf.memory_usage_mb

                # Print to CLI for debugging (RKLLM native stats + deep debug)
                print("--- RKLLM Performance Stats (native) ---")
                print(f"Prefill: {perf.prefill_tokens} tokens in {perf.prefill_time_ms:.2f} ms  ({global_state.prefill_tps:.2f} TPS)")
                print(f"Generate: {perf.generate_tokens} tokens in {perf.generate_time_ms:.2f} ms  ({global_state.generation_tps:.2f} TPS)")
                print(f"Memory Usage: {global_state.memory_usage_mb:.2f} MB")

                # --- Deep Debug -------------------------------------------------
                try:
                    perf_size = ctypes.sizeof(RKLLMPerfStat)
                    perf_bytes = bytes(ctypes.string_at(ctypes.byref(perf), perf_size))
                    print(f"Perf struct size: {perf_size} bytes")
                    print(f"First 32 bytes (hex): {perf_bytes[:32].hex()}")
                except Exception as dbg_e:
                    print(f"[debug] could not dump perf bytes: {dbg_e}")

                print("Token comparison (SDK vs Python):")
                print(f"  Prefill tokens  – SDK: {perf.prefill_tokens}  | Python: {global_state.prompt_word_count}")
                print(f"  Generate tokens – SDK: {perf.generate_tokens} | Python: {global_state.generated_word_count}")
                print("------------------------------------------------------------")
            else:
                # Fallback CLI stats computed in Python
                eval_duration_ms = (global_state.first_token_time - global_state.prompt_eval_start_time) * 1000.0 if global_state.first_token_time > 0 else 0
                gen_duration_ms = (global_state.generation_finish_time - global_state.first_token_time) * 1000.0 if global_state.generation_finish_time > 0 else 0
                if eval_duration_ms > 0:
                    global_state.prefill_tps = global_state.prompt_word_count / (eval_duration_ms/1000.0)
                if gen_duration_ms > 0:
                    global_state.generation_tps = global_state.generated_word_count / (gen_duration_ms/1000.0)
                print("--- RKLLM Performance Stats (fallback) ---")
                print(f"Prefill: {global_state.prompt_word_count} tokens in {eval_duration_ms:.2f} ms  ({global_state.prefill_tps:.2f} TPS)")
                print(f"Generate: {global_state.generated_word_count} tokens in {gen_duration_ms:.2f} ms  ({global_state.generation_tps:.2f} TPS)")


            print("\n", end="", flush=True)
            global_state.finished = True

        elif state == 3:  # Error state (RKLLM_RUN_ERROR)
            print("run error", file=sys.stderr)
            global_state.finished = True
    return

# Connect the callback function between the Python side and the C++ side
callback_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
callback = callback_type(callback_impl)


# ADD THESE FUNCTIONS TO YOUR RKLLM SERVER (after the imports, around line 200):

def get_profession(name: str) -> dict:
    """Returns the profession of a person given their name."""
    people_professions = {
        "john": "Software Engineer",
        "emma": "Doctor", 
        "michael": "Teacher",
        "sarah": "Lawyer",
        "david": "Architect",
        "lisa": "Chef",
        "robert": "Accountant",
        "jennifer": "Marketing Manager",
        "william": "Electrician",
        "olivia": "Graphic Designer"
    }
    
    name_lower = name.lower()
    if name_lower in people_professions:
        return {
            "status": "success",
            "report": f"{name} is a {people_professions[name_lower]}."
        }
    else:
        return {
            "status": "error", 
            "error_message": f"No profession information found for '{name}'."
        }

def get_current_time_string() -> str:
    """Returns the current time in the format 'H:MMam/pm' (e.g., '1:30pm')."""
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%-I:%M%p").lower()

# Built-in tool registry
BUILTIN_TOOLS = {
    'get_profession': get_profession,
    'get_current_time_string': get_current_time_string
}

# REPLACE your existing execute_tool_call function with this:
def execute_tool_call(tool_name, arguments):
    """Execute a tool call and return the result"""
    
    # Check built-in tools first (for Google ADK)
    if tool_name in BUILTIN_TOOLS:
        try:
            result = BUILTIN_TOOLS[tool_name](**arguments)
            return result
        except Exception as e:
            return {"error": f"Built-in tool execution failed: {str(e)}"}
    
    # Fall back to file-based tools (existing functionality)
    if tool_name in TOOL_REGISTRY:
        try:
            result = TOOL_REGISTRY[tool_name](**arguments)
            return result
        except Exception as e:
            return {"error": f"File tool execution failed: {str(e)}"}
    
    return {"error": f"Tool '{tool_name}' not found"}

def parse_tool_calls(text):
    """Parse tool calls from model response"""
    tool_call_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(tool_call_pattern, text, re.DOTALL)
    
    tool_calls = []
    for match in matches:
        try:
            tool_data = json.loads(match)
            tool_call_id = f"call_{str(uuid.uuid4())}"
            tool_calls.append({
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": tool_data.get("name"),
                    "arguments": json.dumps(tool_data.get("arguments", {}))
                }
            })
        except json.JSONDecodeError as e:
            print(f"Error parsing tool call: {e}")
            continue
    
    return tool_calls

def extract_text_without_tool_calls(text):
    """Remove tool call tags from text and return clean content"""
    tool_call_pattern = r'<tool_call>.*?</tool_call>'
    clean_text = re.sub(tool_call_pattern, '', text, flags=re.DOTALL)
    return clean_text.strip()

# Define the RKLLM class
class RKLLM(object):
    def __init__(self, model_path, lora_model_path=None, prompt_cache_path=None):
        rkllm_param = RKLLMParam()
        rkllm_param.model_path = bytes(model_path if model_path else MODEL_PATH, 'utf-8')

        rkllm_param.max_context_len = MAX_CONTEXT_LENGTH
        rkllm_param.max_new_tokens = MAX_NEW_TOKENS
        rkllm_param.skip_special_token = True
        rkllm_param.n_keep = N_KEEP
        
        rkllm_param.top_k = 1
        rkllm_param.top_p = 0.9
        rkllm_param.temperature = 0.8
        rkllm_param.repeat_penalty = 1.1
        rkllm_param.frequency_penalty = 0.0
        rkllm_param.presence_penalty = 0.0
        rkllm_param.mirostat = 0
        rkllm_param.mirostat_tau = 5.0
        rkllm_param.mirostat_eta = 0.1
        rkllm_param.is_async = IS_ASYNC

        rkllm_param.img_start = "".encode('utf-8')
        rkllm_param.img_end = "".encode('utf-8')
        rkllm_param.img_content = "".encode('utf-8')

        rkllm_param.extend_param.base_domain_id = 0
        rkllm_param.extend_param.enabled_cpus_num = CPU_CORE_COUNT
        rkllm_param.extend_param.enabled_cpus_mask = ENABLED_CPU_MASK

        self.handle = RKLLM_Handle_t()

        self.rkllm_init = rkllm_lib.rkllm_init
        self.rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), callback_type]
        self.rkllm_init.restype = ctypes.c_int
        self.rkllm_init(ctypes.byref(self.handle), ctypes.byref(rkllm_param), callback)

        self.rkllm_run = rkllm_lib.rkllm_run
        self.rkllm_run.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
        self.rkllm_run.restype = ctypes.c_int
        
        self.set_chat_template = rkllm_lib.rkllm_set_chat_template
        self.set_chat_template.argtypes = [RKLLM_Handle_t, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        self.set_chat_template.restype = ctypes.c_int

        self.rkllm_destroy = rkllm_lib.rkllm_destroy
        self.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self.rkllm_destroy.restype = ctypes.c_int

        # Handle LoRA adapter if provided
        rkllm_lora_params = None
        if lora_model_path:
            lora_adapter_name = "test"
            lora_adapter = RKLLMLoraAdapter()
            ctypes.memset(ctypes.byref(lora_adapter), 0, ctypes.sizeof(RKLLMLoraAdapter))
            lora_adapter.lora_adapter_path = ctypes.c_char_p((lora_model_path).encode('utf-8'))
            lora_adapter.lora_adapter_name = ctypes.c_char_p((lora_adapter_name).encode('utf-8'))
            lora_adapter.scale = 1.0

            rkllm_load_lora = rkllm_lib.rkllm_load_lora
            rkllm_load_lora.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMLoraAdapter)]
            rkllm_load_lora.restype = ctypes.c_int
            rkllm_load_lora(self.handle, ctypes.byref(lora_adapter))
            rkllm_lora_params = RKLLMLoraParam()
            rkllm_lora_params.lora_adapter_name = ctypes.c_char_p((lora_adapter_name).encode('utf-8'))
        
        self.rkllm_infer_params = RKLLMInferParam()
        ctypes.memset(ctypes.byref(self.rkllm_infer_params), 0, ctypes.sizeof(RKLLMInferParam))
        self.rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
        self.rkllm_infer_params.lora_params = ctypes.pointer(rkllm_lora_params) if rkllm_lora_params else None
        self.rkllm_infer_params.keep_history = KEEP_HISTORY

        # Handle prompt cache if provided
        self.prompt_cache_path = None
        if prompt_cache_path:
            self.prompt_cache_path = prompt_cache_path
            rkllm_load_prompt_cache = rkllm_lib.rkllm_load_prompt_cache
            rkllm_load_prompt_cache.argtypes = [RKLLM_Handle_t, ctypes.c_char_p]
            rkllm_load_prompt_cache.restype = ctypes.c_int
            rkllm_load_prompt_cache(self.handle, ctypes.c_char_p((prompt_cache_path).encode('utf-8')))

    def run(self, prompt):
        rkllm_input = RKLLMInput()
        rkllm_input.input_mode = RKLLMInputMode.RKLLM_INPUT_PROMPT
        rkllm_input.input_data.prompt_input = ctypes.c_char_p(prompt.encode('utf-8'))
        self.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(self.rkllm_infer_params), None)
        return

    def release(self):
        self.rkllm_destroy(self.handle)

def format_messages_to_prompt(messages, tools=None):
    """Convert OpenAI messages format to a prompt string with tool support"""
    prompt_parts = []
    
    # Add system message with tool information
    system_message = None
    for message in messages:
        if message.get('role') == 'system':
            system_message = message.get('content', '')
            break
    
    if system_message:
        if tools:
            # Combine original system message with tool instructions
            tool_instruction = TOOL_SYSTEM_TEMPLATE.render(tools=tools)
            combined_system = f"{system_message}\n\n{tool_instruction}"
            prompt_parts.append(f"System: {combined_system}")
        else:
            prompt_parts.append(f"System: {system_message}")
    elif tools:
        # No system message, but we have tools
        tool_instruction = TOOL_SYSTEM_TEMPLATE.render(tools=tools)
        prompt_parts.append(f"System: {tool_instruction}")
    
    # Add other messages
    for message in messages:
        role = message.get('role', '')
        content = message.get('content', '')
        
        if role == 'system':
            continue  # Already handled above
        elif role == 'user':
            prompt_parts.append(f"User: {content}")
        elif role == 'assistant':
            # Handle assistant messages with tool calls
            if message.get('tool_calls'):
                tool_calls_text = content if content else ""
                for tool_call in message['tool_calls']:
                    func_name = tool_call['function']['name']
                    func_args = tool_call['function']['arguments']
                    tool_calls_text += f"\n<tool_call>\n{{'name': '{func_name}', 'arguments': {func_args}}}\n</tool_call>"
                prompt_parts.append(f"Assistant: {tool_calls_text}")
            else:
                prompt_parts.append(f"Assistant: {content}")
        elif role == 'tool':
            tool_call_id = message.get('tool_call_id', '')
            prompt_parts.append(f"Tool Result: {content}\n\nBased on this tool result, please provide a helpful response to the user. Do not make additional tool calls.")
    
    return "\n".join(prompt_parts) + "\nAssistant:"

def process_conversation_with_tools(messages, tools):
    """Process a conversation with a single tool call iteration"""
    conversation_messages = messages.copy()
    
    # First call: Check if model wants to use tools
    prompt = format_messages_to_prompt(conversation_messages, tools)
    
    # Run the model
    global global_text, global_state
    global_text = []
    global_state = -1
    
    model_thread = threading.Thread(target=rkllm_model.run, args=(prompt,))
    model_thread.start()
    
    full_response = ""
    model_thread_finished = False
    
    while not model_thread_finished:
        while len(global_text) > 0:
            full_response += global_text.pop(0)
            time.sleep(0.005)
        
        model_thread.join(timeout=0.005)
        model_thread_finished = not model_thread.is_alive()
    
    # Check if the response contains tool calls
    tool_calls = parse_tool_calls(full_response)
    
    if tool_calls:
        # Extract text content without tool calls
        content = extract_text_without_tool_calls(full_response)
        
        # Add assistant message with tool calls
        conversation_messages.append({
            "role": "assistant",
            "content": content.strip() if content.strip() else None,
            "tool_calls": tool_calls
        })
        
        # Execute tool calls and add results
        for tool_call in tool_calls:
            func_name = tool_call['function']['name']
            func_args = json.loads(tool_call['function']['arguments'])
            
            # Execute the tool
            tool_result = execute_tool_call(func_name, func_args)
            
            # Add tool result message
            conversation_messages.append({
                "role": "tool",
                "tool_call_id": tool_call['id'],
                "content": json.dumps(tool_result)
            })
        
        # Second call: Get final response after tool execution
        final_prompt = format_messages_to_prompt(conversation_messages, tools)
        final_prompt += "\n\nPlease provide a natural language response based on the tool results above. Do not make any more tool calls."
        
        global_text = []
        global_state = -1
        
        final_thread = threading.Thread(target=rkllm_model.run, args=(final_prompt,))
        final_thread.start()
        
        final_response = ""
        final_thread_finished = False
        
        while not final_thread_finished:
            while len(global_text) > 0:
                final_response += global_text.pop(0)
                time.sleep(0.005)
            
            final_thread.join(timeout=0.005)
            final_thread_finished = not final_thread.is_alive()
        
        return final_response.strip(), conversation_messages
    else:
        # No tool calls, return the response directly
        return full_response.strip(), conversation_messages

# OpenAI API Endpoints

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    global global_state, is_blocking
    
    # Create a lock for the blocking state
    blocking_lock = threading.Lock()
    
    with blocking_lock:
        if is_blocking:
            return openai_error_response(
                "The model is currently busy. Please try again later.",
                error_type="server_error",
                status_code=503
            )
        is_blocking = True
    
    try:
        data = request.json
        if not data:
            return openai_error_response("Missing JSON body")
        
        # Validate required fields
        if 'messages' not in data:
            return openai_error_response("Missing required parameter: messages", param="messages")
        
        messages = data['messages']
        if not isinstance(messages, list) or len(messages) == 0:
            return openai_error_response("Messages must be a non-empty array", param="messages")
        
        # Get other parameters
        model = data.get('model', DEFAULT_MODEL_NAME)
        stream = data.get('stream', False)
        tools = data.get('tools', [])
        tool_choice = data.get('tool_choice')
        max_tokens = data.get('max_tokens')
        temperature = data.get('temperature')
        top_p = data.get('top_p')
        
        lock.acquire()
        try:
            is_blocking = True
            
            # Generate unique ID and timestamp
            completion_id = f"chatcmpl-{str(uuid.uuid4())}"
            created_timestamp = int(datetime.now().timestamp())
            
            if tools:
                # Process conversation with tools (single iteration)
                final_response, final_messages = process_conversation_with_tools(messages, tools)
                
                # Always return the final response (no tool_calls in the final response)
                response = {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created_timestamp,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": final_response
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": count_tokens(str(final_messages)),
                        "completion_tokens": count_tokens(final_response),
                        "total_tokens": count_tokens(str(final_messages)) + count_tokens(final_response)
                    }
                }
                
                return jsonify(response), 200
            else:
                # No tools, use original logic
                prompt = format_messages_to_prompt(messages)
                
                if stream:
                    def generate():
                        global global_state
                        try:
                            # Reset the global state for this new request
                            with global_state.lock:
                                global_state.reset_perf_metrics()
                                global_state.finished = False
                                global_state.prompt_word_count = count_tokens(prompt)
                                global_state.prompt_eval_start_time = time.time()

                            model_thread = threading.Thread(target=rkllm_model.run, args=(prompt,))
                            model_thread.start()
                            
                            full_content = ""
                            
                            while True:
                                try:
                                    # Wait for new content with timeout
                                    chunk = global_state.text_queue.get(timeout=0.1)
                                    full_content += chunk
                                    
                                    chunk_response = {
                                        "id": completion_id,
                                        "object": "chat.completion.chunk",
                                        "created": created_timestamp,
                                        "model": model,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {
                                                "content": chunk
                                            },
                                            "finish_reason": None
                                        }]
                                    }
                                    yield f"data: {json.dumps(chunk_response)}\n\n"
                                    
                                except Exception as e:
                                    # Check if generation is finished
                                    with global_state.lock:
                                        if global_state.finished and global_state.text_queue.empty():
                                            break
                                    
                                    # Check if model thread is still running
                                    if not model_thread.is_alive() and global_state.text_queue.empty():
                                        with global_state.lock:
                                            global_state.finished = True
                                            break
                            
                            # Send final chunk with finish_reason
                            final_chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk", 
                                "created": created_timestamp,
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }]
                            }
                            yield f"data: {json.dumps(final_chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                            
                        except Exception as e:
                            print(f"Error in streaming: {str(e)}")
                            yield f"data: {{'error': 'Stream error: {str(e)}'}}\n\n"
                            yield "data: [DONE]\n\n"
                    
                    return Response(generate(), content_type='text/plain; charset=utf-8')
                
                else:
                    # Non-streaming response
                    global_state = GlobalState()
                    
                    model_thread = threading.Thread(target=rkllm_model.run, args=(prompt,))
                    model_thread.start()
                    
                    full_content = ""
                    
                    while True:
                        try:
                            # Get content with timeout
                            chunk = global_state.text_queue.get(timeout=0.1)
                            full_content += chunk
                        except:
                            # Check if generation is finished
                            with global_state.lock:
                                if global_state.finished and global_state.text_queue.empty():
                                    break
                            
                            # Check if model thread is still running
                            if not model_thread.is_alive() and global_state.text_queue.empty():
                                with global_state.lock:
                                    if global_state.finished:
                                        break
                    
                    response = {
                        "id": completion_id,
                        "object": "chat.completion",
                        "created": created_timestamp,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": full_content.strip()
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": count_tokens(prompt),
                            "completion_tokens": count_tokens(full_content),
                            "total_tokens": count_tokens(prompt) + count_tokens(full_content)
                        }
                    }
                    
                    return jsonify(response), 200
                
        finally:
            lock.release()
            is_blocking = False
            
    except Exception as e:
        return openai_error_response(f"Internal server error: {str(e)}", error_type="server_error", status_code=500)

@app.route('/v1/completions', methods=['POST'])
def completions():
    global global_text, global_state, is_blocking

    if is_blocking or global_state == 0:
        return openai_error_response(
            "The model is currently busy. Please try again later.",
            error_type="server_error",
            status_code=503
        )
    
    try:
        data = request.json
        if not data:
            return openai_error_response("Missing JSON body")
        
        # Validate required fields
        if 'prompt' not in data:
            return openai_error_response("Missing required parameter: prompt", param="prompt")
        
        prompt = data['prompt']
        if not isinstance(prompt, str):
            return openai_error_response("Prompt must be a string", param="prompt")
        
        # Get other parameters
        model = data.get('model', DEFAULT_MODEL_NAME)
        max_tokens = data.get('max_tokens')
        temperature = data.get('temperature')
        top_p = data.get('top_p')
        
        lock.acquire()
        try:
            is_blocking = True
            global_text = []
            global_state = -1
            
            # Generate unique ID and timestamp
            completion_id = f"cmpl-{str(uuid.uuid4())}"
            created_timestamp = int(datetime.now().timestamp())
            
            model_thread = threading.Thread(target=rkllm_model.run, args=(prompt,))
            model_thread.start()
            
            full_completion = ""
            model_thread_finished = False
            
            while not model_thread_finished:
                while len(global_text) > 0:
                    full_completion += global_text.pop(0)
                    time.sleep(0.005)
                
                model_thread.join(timeout=0.005)
                model_thread_finished = not model_thread.is_alive()
            
            response = {
                "id": completion_id,
                "object": "text_completion",
                "created": created_timestamp,
                "model": model,
                "choices": [{
                    "text": full_completion,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": count_tokens(prompt),
                    "completion_tokens": count_tokens(full_completion),
                    "total_tokens": count_tokens(prompt) + count_tokens(full_completion)
                }
            }
            
            return jsonify(response), 200
            
        finally:
            lock.release()
            is_blocking = False
            
    except Exception as e:
        return openai_error_response(f"Internal server error: {str(e)}", error_type="server_error", status_code=500)

# Compatibility route for /v1/ endpoint
@app.route('/luna', methods=['GET'])
def luna_recognition():
    """Device recognition endpoint"""
    return jsonify({"device": "luna"})

# Compatibility route for /v1/ endpoint
@app.route('/v1/', methods=['POST'])
def v1_compatibility():
    """Redirect /v1/ calls to chat completions for compatibility"""
    return chat_completions()

# Models endpoint
@app.route('/v1/models', methods=['GET'])
def models():
    """Return a list of available models mimicking OpenAI's API format"""
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": "luna-large",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "rkllm",
                "permission": [],
                "root": "luna-large",
                "parent": None
            }
        ]
    })

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    with global_state.lock:
        generation_status = "idle" if global_state.finished else "generating"
        response_data = {
            "status": "healthy",
            "generation_status": generation_status,
            "tools_loaded": list(TOOL_REGISTRY.keys()),
            "prefill_speed_tps": f"{global_state.prefill_tps:.2f}",
            "generation_speed_tps": f"{global_state.generation_tps:.2f}",
            "memory_usage_mb": f"{global_state.memory_usage_mb:.2f}"
        }
    return jsonify(response_data), 200

# WiFi connect endpoint
@app.route('/wifi', methods=['POST'])
def wifi_connect():
    """Connect the device to a WiFi network using nmcli.

    Expects JSON with the following structure:
    {
        "uuid": "<WiFi_SSID>",
        "password": "<WiFi_Password>"
    }
    """
    data = request.json
    if not data:
        return openai_error_response("Missing JSON body")

    ssid = data.get('uuid')  # Using the key name specified in the spec
    wifi_password = data.get('password')
    print(wifi_password)
    if not ssid or not wifi_password:
        return openai_error_response("Missing 'uuid' or 'password' parameter", param="uuid/password")

    cmd = ["sudo", "nmcli", "dev", "wifi", "connect", ssid, "password", wifi_password]

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        success = result.returncode == 0
        return jsonify({
            "success": success,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip()
        }), 200 if success else 400
    except Exception as e:
        return openai_error_response(f"Failed to execute nmcli: {str(e)}", error_type="server_error", status_code=500)

# Global model instance
rkllm_model = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rkllm_model_path', type=str, default=LARGE_MODEL_PATH, help='Absolute path of the converted RKLLM model on the Linux board (default from config.py)')
    parser.add_argument('--target_platform', type=str, default=TARGET_PLATFORM, help='Target platform: e.g., rk3588/rk3576 (default from config.py)')
    parser.add_argument('--lora_model_path', type=str, help='Absolute path of the lora_model on the Linux board')
    parser.add_argument('--prompt_cache_path', type=str, help='Absolute path of the prompt_cache file on the Linux board')
    parser.add_argument('--tools_dir', type=str, default='tools', help='Directory containing tool Python files')
    parser.add_argument('--port', type=int, default=SERVER_PORT, help='Port to run the server on (default from config.py)')
    args = parser.parse_args()
    
    print(f"Using model path: {args.rkllm_model_path}")
    print(f"Using target platform: {args.target_platform}")
    print(f"Using tools directory: {args.tools_dir}")

    if not os.path.exists(args.rkllm_model_path):
        print("Error: Please provide the correct rkllm model path, and ensure it is the absolute path on the board.")
        sys.stdout.flush()
        exit()

    if not (args.target_platform in ["rk3588", "rk3576"]):
        print("Error: Please specify the correct target platform: rk3588/rk3576.")
        sys.stdout.flush()
        exit()

    if args.lora_model_path and not os.path.exists(args.lora_model_path):
        print("Error: Please provide the correct lora_model path, and ensure it is the absolute path on the board.")
        sys.stdout.flush()
        exit()

    if args.prompt_cache_path and not os.path.exists(args.prompt_cache_path):
        print("Error: Please provide the correct prompt_cache_file path, and ensure it is the absolute path on the board.")
        sys.stdout.flush()
        exit()


    # Fix frequency
    command = "sudo bash fix_freq_{}.sh".format(args.target_platform)
    subprocess.run(command, shell=True)

    # Set resource limit
    resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))

    # Initialize RKLLM model
    print("=========init....===========")
    sys.stdout.flush()
    model_path = args.rkllm_model_path
    rkllm_model = RKLLM(model_path, args.lora_model_path, args.prompt_cache_path)
    print("RKLLM Model has been initialized successfully!")
    print("OpenAI-compatible API server with tool support is starting...")
    print(f"API Endpoints:")
    print(f"  POST /v1/chat/completions (with tool support)")
    print(f"  POST /v1/completions") 
    print(f"  GET /health")
    print(f"Loaded tools: {list(TOOL_REGISTRY.keys())}")
    print("==============================")
    sys.stdout.flush()

    # Start the Flask application
    app.run(host=SERVER_HOST, port=args.port, threaded=True, debug=DEBUG_MODE)

    print("====================")
    print("RKLLM model inference completed, releasing RKLLM model resources...")
    rkllm_model.release()
    print("====================")