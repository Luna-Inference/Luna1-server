from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Union, Literal
import time
import uuid
import uvicorn

app = FastAPI(title="Luna NPU API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data Models
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    stream: Optional[bool] = False

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    choices: List[ChatCompletionChoice]

class CompletionRequest(BaseModel):
    prompt: Union[str, List[str]]
    stream: Optional[bool] = False

class CompletionChoice(BaseModel):
    text: str
    index: int
    finish_reason: str

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    choices: List[CompletionChoice]

# Simple response models for the new endpoint
class ResponseRequest(BaseModel):
    model: str
    input: str
    stream: Optional[bool] = False

class ContentItem(BaseModel):
    type: str = "output_text"
    text: str
    annotations: List = []

class OutputMessage(BaseModel):
    type: str = "message"
    id: str
    status: str = "completed"
    role: str = "assistant"
    content: List[ContentItem]

class ResponseOutput(BaseModel):
    id: str
    object: str = "response"
    created_at: int
    status: str = "completed"
    error: Optional[str] = None
    model: str
    output: List[OutputMessage]

# Helper functions
def generate_simple_response(input_text: str) -> str:
    """
    Replace this with your actual AI model inference
    This is just a mock response generator for simple input/output
    """
    # Simple mock responses
    mock_responses = [
        f"Here's my response to '{input_text[:50]}...': This is a mock AI response.",
        f"Regarding your input about '{input_text[:30]}...', I would process this with a real AI model.",
        f"Thank you for: '{input_text}'. This is a placeholder response from Luna NPU.",
    ]
    
    import random
    return random.choice(mock_responses)

def generate_response(messages: List[ChatMessage]) -> str:
    """
    Replace this with your actual AI model inference
    This is just a mock response generator
    """
    last_message = messages[-1].content if messages else ""
    
    # Simple mock responses
    mock_responses = [
        f"I understand you said: '{last_message}'. This is a mock response from the AI.",
        f"Thank you for your message about '{last_message[:50]}...'. I'm a placeholder AI.",
        f"Regarding '{last_message}', I would need to process this with a real AI model.",
    ]
    
    import random
    return random.choice(mock_responses)

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Luna NPU API", "version": "1.0.0"}

@app.post("/v1/responses", response_model=ResponseOutput)
async def create_response(request: ResponseRequest):
    try:
        # Generate response using your AI model
        output_text = generate_simple_response(request.input)
        
        response = ResponseOutput(
            id=f"resp_{uuid.uuid4().hex}",
            created_at=int(time.time()),
            model=request.model,
            output=[
                OutputMessage(
                    id=f"msg_{uuid.uuid4().hex}",
                    content=[
                        ContentItem(text=output_text)
                    ]
                )
            ]
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        # Generate response using your AI model
        assistant_response = generate_response(request.messages)
        
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:29]}",
            created=int(time.time()),
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=assistant_response),
                    finish_reason="stop"
                )
            ]
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    try:
        # Handle single prompt or list of prompts
        prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
        
        choices = []
        
        for i, prompt in enumerate(prompts):
            # Generate completion using your AI model
            completion_text = generate_response([ChatMessage(role="user", content=prompt)])
            
            choices.append(
                CompletionChoice(
                    text=completion_text,
                    index=i,
                    finish_reason="stop"
                )
            )
        
        response = CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:29]}",
            created=int(time.time()),
            choices=choices
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
