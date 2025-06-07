#!/usr/bin/env python3

from openai import OpenAI

# Initialize the OpenAI client with your local llama.cpp server
client = OpenAI(
    base_url="http://localhost:1306/v1",  # Point to your local server
    api_key="not-needed"  # llama.cpp often doesn't require a real API key
)

def chat_with_llama(message, model="gpt-3.5-turbo"):
    """
    Send a message to the local llama.cpp server and get a response.
    
    Args:
        message (str): The message to send
        model (str): Model name (can be anything, llama.cpp ignores this usually)
    
    Returns:
        str: The response from the model
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": message}
            ],
            max_tokens=150,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error: {e}"

def main():
    print("llama.cpp Local Chat Client")
    print("-" * 30)
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not user_input:
            continue
            
        print("AI: ", end="", flush=True)
        response = chat_with_llama(user_input)
        print(response)
        print()

if __name__ == "__main__":
    main()