# EchoNode

A modern conversational AI application that uses Ollama to access multiple public models with a clean NiceGUI interface.

## Features

- Integration with Ollama for running AI models locally
- Support for multiple models (Mixtral, Llama, Mistral, Phi, Vicuna)
- Speech recognition and text-to-speech capabilities
- Clean, responsive UI built with NiceGUI
- Comprehensive logging system
- Model switching with a single click

## Setup

1. Install Ollama:
   - Visit [Ollama's website](https://ollama.ai/) to download and install
   - Start the Ollama service

2. Install Python dependencies:
```
pip install nicegui requests speech_recognition pyttsx3 pocketsphinx torch
```

3. Run the application:
```
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:7860
```

## Usage

1. Select a model from the dropdown menu
2. Type your message or use the microphone button to talk
3. Click 'Send' or press Enter to submit your message
4. View the model's response (it will be spoken out loud as well)
5. Pull all available models using the 'Pull All Models' button
6. Expand the Logs section to see detailed interaction logs

## Models

The application uses Ollama to run these models locally:
- **mixtral**: Mixtral 7B model by Mistral AI
- **llama2**: Llama 2 7B model by Meta
- **mistral**: Mistral 7B model
- **phi**: Phi-2 2.7B model by Microsoft
- **vicuna**: Vicuna 7B conversational model

## Logs

Logs are stored in the `logs` directory with timestamps and include:
- User and assistant messages
- Model loading information
- Speech recognition events
- Errors and warnings

## Requirements

- Ollama installed and running locally
- Python 3.7+
- Sufficient RAM for running the models (8GB minimum, 16GB+ recommended)
- Microphone for speech input (optional)
- Speakers for text-to-speech output (optional)
