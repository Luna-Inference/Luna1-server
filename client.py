from openai import OpenAI
from api import *
# Initialize client pointing to your Luna NPU API
client = OpenAI(
    api_key=OPENAI_API_KEY,  # Your API doesn't require authentication
    base_url="http://localhost:8000/v1"
)

response = client.responses.create(
    model="gpt-4.1",
    input="Write a one-sentence bedtime story about a unicorn."
)

# Print the generated text
print("Generated response:")
print(response.output_text)    
# print(response)
