import gradio as gr
from transformers import pipeline
import speech_recognition as sr
import pyttsx3
import os
import torch
import threading

from models import AVAILABLE_MODELS, download_model, get_model_path
from utils.logger import Logger

# Project name
PROJECT_NAME = "EchoNode"

# Create logger
logger = Logger(PROJECT_NAME)

# Default model
DEFAULT_MODEL = "mixtral-7b"

def load_model(model_id):
    """Load a model from the models directory or download if not available"""
    try:
        if model_id not in AVAILABLE_MODELS:
            logger.error(f"Unknown model ID: {model_id}")
            return None
        
        model_info = AVAILABLE_MODELS[model_id]
        model_name = model_info["name"]
        logger.info(f"Loading model {model_name}")
        
        local_path = get_model_path(model_id)
        
        # Check if model exists locally
        if not os.path.exists(local_path):
            logger.info(f"Model {model_name} not found locally, downloading...")
            local_path = download_model(model_id)
        
        # Check for GPU availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load the model
        generator = pipeline('text-generation', model=model_name, device=device)
        logger.info(f"Model {model_name} loaded successfully")
        return generator
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        print(f"Error loading model: {e}. Please ensure model is downloaded.")
        return None

# Initialize STT recognizer
r = sr.Recognizer()

# Initialize TTS engine
engine = pyttsx3.init()

# Initialize model
model = load_model(DEFAULT_MODEL)

# Function from download_models.py, integrated here
def download_all_models():
    """Download all available models and return status updates"""
    results = []
    results.append(f"Available models: {list(AVAILABLE_MODELS.keys())}")
    
    for model_id in AVAILABLE_MODELS:
        results.append(f"\nDownloading model: {model_id}")
        try:
            download_model(model_id)
            results.append(f"Model {model_id} downloaded successfully.")
        except Exception as e:
            error_msg = f"Error downloading model {model_id}: {e}"
            logger.error(error_msg)
            results.append(error_msg)
    
    results.append("\nAll models processed!")
    return "\n".join(results)

def download_models_background(progress=None):
    """Download models in a background thread with optional progress updates"""
    def download_thread():
        result = download_all_models()
        if progress is not None:
            progress(result, "")
    
    thread = threading.Thread(target=download_thread)
    thread.daemon = True
    thread.start()
    return "Download started in background. See progress below."

# Function to generate response from LLM
def generate_response(message, history, current_log, model_id):
    """Generate a response from the model"""
    global model
    
    # Update log with user message
    current_log = logger.log_user_message(message)
    
    # Check if we need to switch models
    current_model_id = getattr(generate_response, "current_model_id", DEFAULT_MODEL)
    if model_id != current_model_id:
        logger.info(f"Switching model from {current_model_id} to {model_id}")
        model = load_model(model_id)
        generate_response.current_model_id = model_id
    
    try:
        # If model is not loaded successfully, return error message
        if model is None:
            response = "Sorry, the model couldn't be loaded. Please check the logs for more information."
        else:
            # Prepare input for LLM with conversation history in Mistral's instruction format
            prompt = "<s>[INST] "
            for user_msg, assistant_msg in history:
                prompt += f"User: {user_msg}\nAssistant: {assistant_msg}\n"
            prompt += f"User: {message}\n[/INST]"
            
            # Generate response
            result = model(prompt, max_length=500, do_sample=True, temperature=0.7)
            response = result[0]['generated_text'].split("[/INST]")[-1].strip()
            if not response:
                response = "I didn't generate a response. Try again!"
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        response = f"Error generating response: {e}"
    
    # Update log with assistant response
    current_log = logger.log_assistant_message(response)
    
    # Schedule TTS for the response
    try:
        engine.say(response)
        engine.runAndWait()
    except Exception as e:
        logger.error(f"TTS error: {e}")
    
    return response, current_log

# Function to handle speech input
def speech_to_text():
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source, timeout=5)
    try:
        user_text = r.recognize_sphinx(audio)
        logger.info(f"Recognized speech: {user_text}")
        return user_text
    except sr.UnknownValueError:
        return "Sorry, I didn't catch that."
    except sr.RequestError as e:
        logger.error(f"STT error: {e}")
        return f"STT error: {e}"

# Gradio interface setup
with gr.Blocks(title=PROJECT_NAME) as demo:
    # State to maintain logs
    log_state = gr.State(value=logger.get_chat_history())
    
    # Current model ID
    current_model_id = gr.State(value=DEFAULT_MODEL)
    
    # Header
    gr.Markdown(f"# {PROJECT_NAME} - Local Chatbot")
    
    with gr.Row():
        # Model selection dropdown
        model_dropdown = gr.Dropdown(
            choices=list(AVAILABLE_MODELS.keys()),
            value=DEFAULT_MODEL,
            label="Select Model",
            info="Choose which AI model to use"
        )
        
        # Download all models button
        download_button = gr.Button("Download All Models", variant="secondary")
    
    # Download progress display
    download_status = gr.Textbox(label="Download Status", lines=10, visible=True)
    
    gr.Markdown("Speak or type to chat. Logs are below.")
    
    # Chat interface
    chat = gr.ChatInterface(
        fn=generate_response,
        additional_inputs=[log_state, model_dropdown],
        additional_outputs=[log_state],
        title="Chat with EchoNode",
        description="A local chatbot powered by multiple LLM models, running on your GPU."
    )
    
    # Speech input button
    with gr.Row():
        speech_button = gr.Button("Speak", variant="primary")
        speech_output = gr.Textbox(label="Speech Input", placeholder="Click 'Speak' to talk")
    
    # Log display
    log_display = gr.Markdown(label="Logs", value=log_state.value)
    
    # Update log display when log_state changes
    log_state.change(
        fn=lambda x: x,
        inputs=[log_state],
        outputs=[log_display]
    )
    
    # Update model when dropdown changes
    model_dropdown.change(
        fn=lambda x: x,
        inputs=[model_dropdown], 
        outputs=[current_model_id]
    )
    
    # Connect download button to download function
    download_button.click(
        fn=download_models_background,
        outputs=[download_status]
    )
    
    # Event handlers
    speech_button.click(fn=speech_to_text, outputs=speech_output).then(
        fn=lambda x: x, inputs=speech_output, outputs=chat.textbox
    )

# Launch the application
if __name__ == "__main__":
    demo.launch(server_name="localhost", server_port=7860)