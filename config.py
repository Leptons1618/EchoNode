"""
Application configuration constants
"""

# Project information
PROJECT_NAME = "EchoNode"

# Model configuration
DEFAULT_MODEL = "mixtral"

# Available models
AVAILABLE_MODELS = {
    "llama2": {
        "name": "llama2",
        "description": "Llama 2 7B model from Meta"
    },
    "mixtral": {
        "name": "mixtral",
        "description": "Mixtral 7B Instruct model from Mistral AI"
    },
    "mistral": {
        "name": "mistral",
        "description": "Mistral 7B Instruct model"
    },
    "phi": {
        "name": "phi",
        "description": "Phi-2 2.7B model from Microsoft"
    },
    "vicuna": {
        "name": "vicuna",
        "description": "Vicuna 7B conversational model"
    }
}

# UI Configuration
UI_COLORS = {
    "primary": "#4f46e5",  # Indigo color for primary elements
    "background": "#1a1a1a"  # Dark background for dark mode
}

# UI Dimensions
UI_DIMENSIONS = {
    "width": "1000px",       # Width of the main content
    "chat_height": "500px",  # Height of the chat container
    "input_height": "80px"   # Height of the input area
}

# Generation parameters
GENERATION_PARAMS = {
    "temperature": 0.7,
    "max_tokens": 500
}
