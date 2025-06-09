from openai import OpenAI
# from api import *
# Configure client to use our local server
client = OpenAI(
    base_url="http://localhost:8080/v1",  # Point to our local server
    api_key = "sk-rkllm-api-key",
    #api_key=OPENAI_API_KEY          # Use the API key defined in openai_server.py
)

completion = client.chat.completions.create(
    model="gpt-4.1",  # Model name doesn't matter, our server ignores it
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Write a one-sentence bedtime story about a unicorn."
        }
    ]
)

# Print the response
print(completion.choices[0].message.content)