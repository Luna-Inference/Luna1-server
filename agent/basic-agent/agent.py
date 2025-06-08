import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
import litellm
litellm._turn_on_debug()

def get_profession(name: str) -> dict:
    """Returns the profession of a person given their name.

    Args:
        name (str): The name of the person to look up.

    Returns:
        dict: status and result or error msg.
    """
    # Dictionary containing 10 people and their professions
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
    
    # Convert name to lowercase for case-insensitive comparison
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


root_agent = Agent(
    name="profession_agent",
    # model= LiteLlm(model="ollama_chat/qwen3:0.6b"),
    model = LiteLlm(model="openai/mistral-small3.1"),
    description=(
        "Agent to answer questions people's professions."
    ),
    instruction=(
        "You are a helpful agent who can answer user questions about people's professions."
    ),
    tools=[get_profession],
)