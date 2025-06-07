from openai import OpenAI

# Initialize client pointing to your Luna NPU API
client = OpenAI(
    api_key="dummy-key",  # Your API doesn't require authentication
    base_url="http://localhost:8000/v1"
)

# Test response completion
stream = client.responses.create(
    model="gpt-4.1",
    input="Hello.",
    stream=True,
    
)

for chunk in stream:
    print(chunk)
    print(chunk.choices[0].delta)


