# EchoNode

A conversational AI application that uses multiple public models for text generation with voice interaction capabilities.

## Features

- Multiple model support (Mixtral, Llama, Vicuna, Phi)
- Speech recognition and text-to-speech capabilities
- Comprehensive logging system with file storage
- User-friendly interface with Gradio
- Model switching within the chat interface
- Background downloading of models with status updates

## Setup

1. Install dependencies:
```
pip install transformers torch gradio speech_recognition pyttsx3 pocketsphinx
```

2. Run the application:
```
python app.py
```

3. Use the "Download All Models" button in the interface to download models directly from the UI, or select individual models from the dropdown menu to download them as needed.

## Usage

1. Select a model from the dropdown menu
2. Type your message or use the "Speak" button to talk
3. View the model's response (it will be spoken out loud as well)
4. Monitor logs below the chat interface
5. Download all available models using the "Download All Models" button

## Models

- **mixtral-7b**: Mixtral 7B Instruct model from Mistral AI
- **llama-7b**: Llama 2 7B Chat model from Meta
- **vicuna-7b**: Vicuna 7B conversational model
- **phi-2**: Phi-2 2.7B model from Microsoft

## Logs

Logs are stored in the `logs` directory with timestamps and include:
- User and assistant messages
- Model loading information
- Speech recognition events
- Errors and warnings

## Model Storage

Models are downloaded and stored in the models directory. The structure is:
- `models/mixtral-7b/` - Mixtral model files
- `models/llama-7b/` - Llama model files
- `models/vicuna-7b/` - Vicuna model files
- `models/phi-2/` - Phi-2 model files

## Development

The application is built with:
- Gradio for the web interface
- Transformers for the language models
- PyTorch as the underlying machine learning framework
- Speech recognition and text-to-speech libraries for voice interaction
