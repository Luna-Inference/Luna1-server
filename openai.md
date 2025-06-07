# Essential OpenAI API Endpoints

This document lists the core OpenAI API endpoints that may be used as reference for implementing compatible interfaces.

## Authentication

All API requests require authentication using an API key in the Authorization header:
```
Authorization: Bearer YOUR_API_KEY
```

## Base URL

Production API: `https://api.openai.com/v1/`

## Chat Completions

### Create Chat Completion
**POST** `/v1/chat/completions`

- Used for ChatGPT-style conversations
- Input: Messages array with role/content pairs
- Output: Response with model-generated message

**Request Body:**
```json
{
  "model": "gpt-3.5-turbo",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "stream": false
}
```

**Response:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677858242,
  "model": "gpt-3.5-turbo",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help you today?"
    },
    "index": 0
  }]
}
```

## Text Completions

### Create Completion
**POST** `/v1/completions`

- The legacy endpoint for text generation
- Takes a prompt string and returns a completion

**Request Body:**
```json
{
  "model": "text-davinci-003",
  "prompt": "Say this is a test",
  "max_tokens": 100
}
```

**Response:**
```json
{
  "id": "cmpl-123",
  "object": "text_completion",
  "created": 1589478378,
  "model": "text-davinci-003",
  "choices": [{
    "text": "\n\nThis is a test",
    "index": 0
  }]
}
```
