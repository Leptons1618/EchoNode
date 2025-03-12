import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define available models
AVAILABLE_MODELS = {
    "mixtral-7b": {
        "name": "mistralai/Mixtral-7B-Instruct-v0.2",
        "description": "Mixtral 7B Instruct model from Mistral AI"
    },
    "llama-7b": {
        "name": "meta-llama/Llama-2-7b-chat-hf",
        "description": "Llama 2 7B Chat model from Meta"
    },
    "vicuna-7b": {
        "name": "lmsys/vicuna-7b-v1.5",
        "description": "Vicuna 7B conversational model"
    },
    "phi-2": {
        "name": "microsoft/phi-2",
        "description": "Phi-2 2.7B model from Microsoft"
    }
}

def get_model_path(model_id):
    """Returns the local path for a given model ID"""
    models_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(models_dir, model_id)

def download_model(model_id):
    """Downloads and saves a model from HuggingFace to local storage"""
    if model_id not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model ID: {model_id}")
    
    model_info = AVAILABLE_MODELS[model_id]
    model_name = model_info["name"]
    local_path = get_model_path(model_id)
    
    print(f"Downloading model {model_name} to {local_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(local_path, exist_ok=True)
    
    # Download model and tokenizer
    AutoModelForCausalLM.from_pretrained(model_name, cache_dir=local_path)
    AutoTokenizer.from_pretrained(model_name, cache_dir=local_path)
    
    print(f"Model {model_name} downloaded successfully")
    return local_path
