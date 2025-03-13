import os
import torch
import asyncio
import requests
import threading
import queue
import time
import json
import psutil
import gc
import speech_recognition as sr
import pyttsx3
from datetime import datetime
from nicegui import ui, app, Client
from dotenv import load_dotenv
from utils.logger import Logger
from config import PROJECT_NAME, DEFAULT_MODEL, AVAILABLE_MODELS, UI_COLORS, GENERATION_PARAMS, UI_DIMENSIONS
from aiohttp import ClientSession, ClientTimeout
from concurrent.futures import ThreadPoolExecutor

# Load environment variables from .env file
load_dotenv()

# Get config from environment variables
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434/api")
HOST = os.getenv("HOST", "localhost")
PORT = int(os.getenv("PORT", 7860))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 5))
MAX_HISTORY = int(os.getenv("MAX_HISTORY", 10))
RESPONSE_CACHE_SIZE = int(os.getenv("RESPONSE_CACHE_SIZE", 100))

# Create logger
logger = Logger(PROJECT_NAME)

# Create thread pool executor
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Initialize STT recognizer
r = sr.Recognizer()

# Initialize TTS engine
engine = pyttsx3.init()

# Chat history
conversation_history = []

# Response cache
response_cache = {}

# System metrics
system_metrics = {
    'cpu': 0,
    'memory': 0,
    'gpu_memory': 0,
}

# Initialize AIOHTTP session
session = None

async def initialize_session():
    """Initialize global aiohttp session"""
    global session
    if session is None or session.closed:
        session = ClientSession(timeout=ClientTimeout(total=120))
    return session

async def check_ollama_status():
    """Check if Ollama is running and accessible using async"""
    s = await initialize_session()
    try:
        async with s.get(f"{OLLAMA_API_BASE}/tags") as response:
            if response.status == 200:
                json_response = await response.json()
                available_models = json_response.get('models', [])
                model_names = [m['name'] for m in available_models]
                logger.info(f"Ollama is running. Available models: {model_names}")
                return True, available_models
            else:
                logger.error(f"Ollama returned status code: {response.status}")
                return False, []
    except Exception as e:
        logger.error(f"Error connecting to Ollama: {e}")
        return False, []

def truncate_conversation_history():
    """Truncate conversation history to prevent memory issues"""
    global conversation_history
    if len(conversation_history) > MAX_HISTORY * 2:  # Each exchange has 2 entries (user+assistant)
        # Keep first message for context and last MAX_HISTORY-1 exchanges
        conversation_history = [conversation_history[0]] + conversation_history[-(MAX_HISTORY*2-1):]
        logger.info(f"Truncated conversation history to {len(conversation_history)} messages")

def cleanup_memory():
    """Force garbage collection to free memory"""
    collected = gc.collect()
    logger.info(f"Garbage collector: collected {collected} objects.")

async def pull_ollama_model(model_name, status_label):
    """Pull a model from Ollama asynchronously"""
    s = await initialize_session()
    
    status_label.text = f"Pulling model {model_name}..."
    
    try:
        async with s.post(
            f"{OLLAMA_API_BASE}/pull",
            json={"name": model_name, "stream": True}
        ) as response:
            if response.status == 200:
                # Process streaming response
                status_update = ""
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line)
                            if 'status' in data:
                                status_update = f"Status: {data['status']}"
                                if 'completed' in data and 'total' in data:
                                    percentage = (data['completed'] / data['total']) * 100 if data['total'] > 0 else 0
                                    status_update += f" - {percentage:.1f}% ({data['completed']}/{data['total']})"
                                status_label.text = f"Pulling {model_name}: {status_update}"
                                await asyncio.sleep(0.1)  # Small sleep to prevent UI overload
                        except json.JSONDecodeError:
                            pass
                
                status_label.text = f"Model {model_name} pulled successfully!"
                return True
            else:
                status_label.text = f"Error pulling model: {response.status}"
                return False
    except Exception as e:
        status_label.text = f"Error pulling model: {e}"
        logger.error(f"Error pulling model {model_name}: {e}")
        return False

def get_cache_key(model_name, message, history_summary):
    """Generate a cache key for responses"""
    # Create a hash based on model, message and recent history
    key = f"{model_name}:{message}:{history_summary}"
    return hash(key)

def update_system_metrics():
    """Update system metrics in a separate thread"""
    while True:
        try:
            # CPU and memory usage
            system_metrics['cpu'] = psutil.cpu_percent()
            system_metrics['memory'] = psutil.virtual_memory().percent
            
            # GPU usage if available
            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                gpu_memory = torch.cuda.memory_allocated(current_device) / torch.cuda.max_memory_allocated(current_device) * 100 if torch.cuda.max_memory_allocated(current_device) > 0 else 0
                system_metrics['gpu_memory'] = gpu_memory
            
            time.sleep(1)  # Update every second
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
            time.sleep(5)  # Wait longer on error

async def generate_response_streaming(message, model_name, chat_container, status_element):
    """Generate a response using Ollama API with streaming"""
    global conversation_history, response_cache
    
    try:
        # Log user message
        logger.log_user_message(message)
        
        # Update conversation history
        conversation_history.append({"role": "user", "content": message})
        
        # Truncate history if too long
        truncate_conversation_history()
        
        # Get recent history summary for caching
        history_summary = ""
        if len(conversation_history) >= 4:
            # Use last 2 exchanges for cache key
            history_summary = "".join([msg["content"][:20] for msg in conversation_history[-4:]])
        
        # Check cache
        cache_key = get_cache_key(model_name, message, history_summary)
        if cache_key in response_cache:
            logger.info("Using cached response")
            assistant_response = response_cache[cache_key]
            
            # Add to history
            conversation_history.append({"role": "assistant", "content": assistant_response})
            
            # Log assistant response
            logger.log_assistant_message(assistant_response)
            
            # Text to speech in a separate thread to not block the UI
            executor.submit(lambda: speak_text(assistant_response))
            
            # Update UI and return
            with chat_container:
                ui.chat_message(assistant_response, name='EchoNode').classes('row-reverse')
            status_element.text = ""
            return assistant_response
        
        # Prepare request for Ollama
        payload = {
            "model": model_name,
            "messages": conversation_history,
            "stream": True,
            "options": {
                "temperature": GENERATION_PARAMS["temperature"],
                "num_predict": GENERATION_PARAMS["max_tokens"]
            }
        }
        
        # Update status
        status_element.text = "Generating response..."
        
        # Make API call
        s = await initialize_session()
        async with s.post(f"{OLLAMA_API_BASE}/chat", json=payload) as response:
            if response.status == 200:
                # For streaming responses
                full_response = ""
                message_ui = None
                
                # Create a placeholder for the streaming response
                with chat_container:
                    message_ui = ui.chat_message("", name='EchoNode').classes('row-reverse')
                
                async for line in response.content:
                    if line:
                        try:
                            chunk = json.loads(line)
                            if chunk.get('message', {}).get('content'):
                                content = chunk['message']['content']
                                full_response += content
                                # Update UI
                                message_ui.text = full_response
                                await asyncio.sleep(0.01)  # Small sleep to prevent UI overload
                        except json.JSONDecodeError:
                            pass
                
                # Update conversation history with full response
                conversation_history.append({"role": "assistant", "content": full_response})
                
                # Cache the response
                if len(response_cache) >= RESPONSE_CACHE_SIZE:
                    # Remove a random key to prevent cache from growing too large
                    response_cache.pop(next(iter(response_cache)))
                response_cache[cache_key] = full_response
                
                # Log assistant response
                logger.log_assistant_message(full_response)
                
                # Text to speech in a separate thread
                executor.submit(lambda: speak_text(full_response))
                
                # Clear status
                status_element.text = ""
                
                # Run cleanup in the background
                asyncio.create_task(run_async_cleanup())
                
                return full_response
            else:
                error_msg = f"Error: API returned status code {response.status}"
                logger.error(error_msg)
                status_element.text = error_msg
                with chat_container:
                    ui.chat_message(error_msg, name='EchoNode').classes('row-reverse text-red-600')
                return error_msg
    except Exception as e:
        error_msg = f"Error generating response: {e}"
        logger.error(error_msg)
        status_element.text = error_msg
        with chat_container:
            ui.chat_message(error_msg, name='EchoNode').classes('row-reverse text-red-600')
        return error_msg

async def run_async_cleanup():
    """Run cleanup tasks asynchronously"""
    await asyncio.sleep(0.1)  # Small delay to let other tasks finish
    # Run garbage collection in a separate thread to avoid blocking
    await asyncio.to_thread(cleanup_memory)

def speak_text(text):
    """TTS function that can be run in a separate thread"""
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        logger.error(f"TTS error: {e}")

async def speech_to_text_async(mic_button, user_input, status):
    """Convert speech to text asynchronously"""
    mic_button.disable()
    status.text = "Listening..."
    
    try:
        # Run speech recognition in a thread to not block the UI
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(executor, speech_to_text_worker)
        
        if result.startswith("Error:"):
            status.text = result
        elif result == "Sorry, I didn't catch that.":
            status.text = result
        else:
            user_input.value = result
            status.text = ""
            logger.info(f"Recognized speech: {result}")
    except Exception as e:
        status.text = f"Error: {e}"
        logger.error(f"Speech recognition error: {e}")
    
    mic_button.enable()

def speech_to_text_worker():
    """Worker function for speech recognition to be run in a separate thread"""
    try:
        with sr.Microphone() as source:
            audio = r.listen(source, timeout=5)
        try:
            text = r.recognize_sphinx(audio)
            return text
        except sr.UnknownValueError:
            return "Sorry, I didn't catch that."
        except sr.RequestError as e:
            return f"Error: STT error: {e}"
    except Exception as e:
        return f"Error: {e}"

async def download_all_models(status_label):
    """Pull all available models from Ollama asynchronously"""
    status_label.text = "Starting to pull models..."
    
    for model_id, model_info in AVAILABLE_MODELS.items():
        model_name = model_info["name"]
        await pull_ollama_model(model_name, status_label)
    
    status_label.text += "\nAll model pulls completed."

async def create_ui():
    """Create NiceGUI user interface asynchronously"""
    # Check Ollama status
    ollama_running, available_models = await check_ollama_status()
    
    # Start metrics collection thread
    metrics_thread = threading.Thread(target=update_system_metrics, daemon=True)
    metrics_thread.start()
    
    with ui.card().classes('w-full'):
        ui.label(f"{PROJECT_NAME} - Local Chatbot").classes('text-2xl font-bold text-center')
        
        # System metrics display
        with ui.row().classes('w-full'):
            cpu_indicator = ui.linear_progress(0).props('size=xs color=primary')
            cpu_label = ui.label('CPU: 0%').classes('text-xs')
            mem_indicator = ui.linear_progress(0).props('size=xs color=warning')
            mem_label = ui.label('MEM: 0%').classes('text-xs')
            if torch.cuda.is_available():
                gpu_indicator = ui.linear_progress(0).props('size=xs color=accent')
                gpu_label = ui.label('GPU: 0%').classes('text-xs')
        
        # Model selection and download section
        with ui.row().classes('w-full justify-between items-center'):
            model_selector = ui.select(
                options=[{"label": f"{model_info['name']} - {model_info['description']}", 
                          "value": model_id} for model_id, model_info in AVAILABLE_MODELS.items()],
                value=DEFAULT_MODEL,
                label="Select Model"
            ).classes('w-2/3')
            
            download_button = ui.button('Pull All Models', icon='download')
            download_status = ui.label('')
            
            download_button.on('click', lambda: download_all_models(download_status))
        
        # Status messages
        status_element = ui.label().classes('text-amber-600')
        if not ollama_running:
            status_element.text = "Ollama is not running! Please start Ollama server first."
            status_element.classes('text-red-600 font-bold')
        
        # Chat interface - use dimensions from config
        chat_container = ui.card().classes('w-full overflow-y-auto').style(f'height: {UI_DIMENSIONS["chat_height"]}')
        
        # Input area - use dimensions from config
        with ui.row().classes('w-full gap-2 items-end').style(f'min-height: {UI_DIMENSIONS["input_height"]}'):
            user_input = ui.input(label='Message', placeholder='Type your message here').classes('w-full')
            send_button = ui.button('Send', icon='send').props('color=primary')
            mic_button = ui.button(icon='mic')
        
        # Log display
        with ui.expansion('Logs', icon='article').classes('w-full'):
            log_display = ui.markdown().classes('w-full')
        
        async def send_message():
            if not user_input.value:
                return
            
            message = user_input.value
            user_input.value = ''
            
            # Add user message to chat
            with chat_container:
                ui.chat_message(message, name='You')
            
            # Get selected model
            model_name = AVAILABLE_MODELS[model_selector.value]['name']
            
            # Generate response with streaming
            await generate_response_streaming(message, model_name, chat_container, status_element)
            
            # Update log display
            log_display.content = logger.get_chat_history()
        
        # Update system metrics every second
        async def update_ui_metrics():
            while True:
                cpu_indicator.value = system_metrics['cpu'] / 100
                cpu_label.text = f"CPU: {system_metrics['cpu']:.1f}%"
                
                mem_indicator.value = system_metrics['memory'] / 100
                mem_label.text = f"MEM: {system_metrics['memory']:.1f}%"
                
                if torch.cuda.is_available():
                    gpu_indicator.value = system_metrics['gpu_memory'] / 100
                    gpu_label.text = f"GPU: {system_metrics['gpu_memory']:.1f}%"
                
                await asyncio.sleep(1)
        
        # Start updating metrics
        asyncio.create_task(update_ui_metrics())
        
        # Button handlers
        send_button.on('click', send_message)
        user_input.on('keydown.enter', send_message)
        mic_button.on('click', lambda: speech_to_text_async(mic_button, user_input, status_element))
        
        # Initialize log display
        log_display.content = logger.get_chat_history()

if __name__ == "__main__":
    # Close any existing aiohttp session when restarting
    async def cleanup():
        global session
        if session and not session.closed:
            await session.close()
    
    app.on_shutdown(cleanup)
    
    # Configure NiceGUI
    app.title = PROJECT_NAME
    ui.colors(primary=UI_COLORS["primary"])
    
    # CSS customization
    ui.add_head_html(f'''
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }}
            .nicegui-content {{
                max-width: {UI_DIMENSIONS["width"]};
                margin: 0 auto;
                padding: 2rem;
            }}
            .streaming-text {{
                transition: all 0.1s ease;
            }}
        </style>
    ''')
    
    # Run with dark mode and native mode enabled
    ui.run(
        title=PROJECT_NAME, 
        host=HOST, 
        port=PORT, 
        on_startup=create_ui,
        dark=True,   # Enable dark mode
        native=True  # Enable native mode
    )