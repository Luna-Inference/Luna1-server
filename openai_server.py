import ctypes
import sys
import os
import subprocess
import resource
import threading
import time
import argparse
import json
from flask import Flask, request, jsonify, Response
from config import *
import uuid
from datetime import datetime

app = Flask(__name__)

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

class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
        ("logits", RKLLMResultLogits)
    ]

# OpenAI API Configuration
DEFAULT_MODEL_NAME = "rkllm-local"  # Default model name to return

# Create a lock to control multi-user access to the server.
lock = threading.Lock()

# Create a global variable to indicate whether the server is currently in a blocked state.
is_blocking = False

# Define global variables to store the callback function output
global_text = []
global_state = -1
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
    global global_text, global_state, split_byte_data
    if state == LLMCallState.RKLLM_RUN_FINISH:
        global_state = state
        print("\n")
        sys.stdout.flush()
    elif state == LLMCallState.RKLLM_RUN_ERROR:
        global_state = state
        print("run error")
        sys.stdout.flush()
    elif state == LLMCallState.RKLLM_RUN_NORMAL:
        global_state = state
        if result.contents.text:
            text_chunk = result.contents.text.decode('utf-8')
            global_text.append(text_chunk)

# Connect the callback function between the Python side and the C++ side
callback_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
callback = callback_type(callback_impl)

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

def format_messages_to_prompt(messages):
    """Convert OpenAI messages format to a single prompt string"""
    prompt_parts = []
    for message in messages:
        role = message.get('role', '')
        content = message.get('content', '')
        
        if role == 'system':
            prompt_parts.append(f"System: {content}")
        elif role == 'user':
            prompt_parts.append(f"User: {content}")
        elif role == 'assistant':
            prompt_parts.append(f"Assistant: {content}")
    
    return "\n".join(prompt_parts) + "\nAssistant:"

# OpenAI API Endpoints

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
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
        if 'messages' not in data:
            return openai_error_response("Missing required parameter: messages", param="messages")
        
        messages = data['messages']
        if not isinstance(messages, list) or len(messages) == 0:
            return openai_error_response("Messages must be a non-empty array", param="messages")
        
        # Get other parameters
        model = data.get('model', DEFAULT_MODEL_NAME)
        stream = data.get('stream', False)
        max_tokens = data.get('max_tokens')
        temperature = data.get('temperature')
        top_p = data.get('top_p')
        
        lock.acquire()
        try:
            is_blocking = True
            global_text = []
            global_state = -1
            
            # Convert messages to prompt
            prompt = format_messages_to_prompt(messages)
            
            # Generate unique ID and timestamp
            completion_id = f"chatcmpl-{str(uuid.uuid4())}"
            created_timestamp = int(datetime.now().timestamp())
            
            if stream:
                def generate():
                    try:
                        model_thread = threading.Thread(target=rkllm_model.run, args=(prompt,))
                        model_thread.start()
                        
                        full_content = ""
                        model_thread_finished = False
                        
                        while not model_thread_finished:
                            while len(global_text) > 0:
                                chunk_content = global_text.pop(0)
                                full_content += chunk_content
                                
                                chunk_response = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created_timestamp,
                                    "model": model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {
                                            "content": chunk_content
                                        },
                                        "finish_reason": None
                                    }]
                                }
                                yield f"data: {json.dumps(chunk_response)}\n\n"
                            
                            model_thread.join(timeout=0.005)
                            model_thread_finished = not model_thread.is_alive()
                        
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
                        
                    finally:
                        pass
                
                return Response(generate(), content_type='text/plain; charset=utf-8')
            
            else:
                # Non-streaming response
                model_thread = threading.Thread(target=rkllm_model.run, args=(prompt,))
                model_thread.start()
                
                full_content = ""
                model_thread_finished = False
                
                while not model_thread_finished:
                    while len(global_text) > 0:
                        full_content += global_text.pop(0)
                        time.sleep(0.005)
                    
                    model_thread.join(timeout=0.005)
                    model_thread_finished = not model_thread.is_alive()
                
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
                        "prompt_tokens": len(prompt.split()),  # Rough estimation
                        "completion_tokens": len(full_content.split()),  # Rough estimation
                        "total_tokens": len(prompt.split()) + len(full_content.split())
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
                    "prompt_tokens": len(prompt.split()),  # Rough estimation
                    "completion_tokens": len(full_completion.split()),  # Rough estimation  
                    "total_tokens": len(prompt.split()) + len(full_completion.split())
                }
            }
            
            return jsonify(response), 200
            
        finally:
            lock.release()
            is_blocking = False
            
    except Exception as e:
        return openai_error_response(f"Internal server error: {str(e)}", error_type="server_error", status_code=500)

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

# Global model instance
rkllm_model = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rkllm_model_path', type=str, default=MODEL_PATH, help='Absolute path of the converted RKLLM model on the Linux board (default from config.py)')
    parser.add_argument('--target_platform', type=str, default=TARGET_PLATFORM, help='Target platform: e.g., rk3588/rk3576 (default from config.py)')
    parser.add_argument('--lora_model_path', type=str, help='Absolute path of the lora_model on the Linux board')
    parser.add_argument('--prompt_cache_path', type=str, help='Absolute path of the prompt_cache file on the Linux board')
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on')
    args = parser.parse_args()
    
    print(f"Using model path: {args.rkllm_model_path}")
    print(f"Using target platform: {args.target_platform}")

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
    print("OpenAI-compatible API server is starting...")
    print(f"API Endpoints:")
    print(f"  POST /v1/chat/completions")
    print(f"  POST /v1/completions") 
    print(f"  GET /health")
    print("==============================")
    sys.stdout.flush()

    # Start the Flask application
    app.run(host=SERVER_HOST, port=args.port, threaded=True, debug=DEBUG_MODE)

    print("====================")
    print("RKLLM model inference completed, releasing RKLLM model resources...")
    rkllm_model.release()
    print("====================")