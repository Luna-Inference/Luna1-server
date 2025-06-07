import openai
from api import *
client = openai.OpenAI(
    api_key="",
    base_url="http://localhost:8080/v1"
)

# Define your tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# Make request with tools
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {"role": "user", "content": "What's the weather in Paris?"}
    ],
    tools=tools,
    tool_choice="auto"
)

# Check if model wants to call a function
if response.choices[0].finish_reason == "tool_calls":
    print("Model wants to call:", response.choices[0].message.tool_calls)