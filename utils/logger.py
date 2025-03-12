import os
import logging
import datetime

class Logger:
    def __init__(self, project_name, log_dir="logs"):
        self.project_name = project_name
        
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure file logger
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{project_name}_{timestamp}.log")
        
        # Configure logging
        self.logger = logging.getLogger(project_name)
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(file_formatter)
        self.logger.addHandler(console_handler)
        
        self.chat_history = f"[{project_name}] Started at {timestamp} on {os.getlogin()}'s machine\n"
        self.info(f"Session started")
    
    def info(self, message):
        """Log an info message"""
        self.logger.info(message)
    
    def warning(self, message):
        """Log a warning message"""
        self.logger.warning(message)
    
    def error(self, message):
        """Log an error message"""
        self.logger.error(message)
    
    def log_user_message(self, message):
        """Log a user message to the chat history"""
        self.info(f"User: {message}")
        self.chat_history += f"[{self.project_name}] User: {message}\n"
        return self.chat_history
    
    def log_assistant_message(self, message):
        """Log an assistant message to the chat history"""
        self.info(f"Assistant: {message[:50]}...")  # Log start of message to file
        self.chat_history += f"[{self.project_name}] Assistant: {message}\n"
        return self.chat_history
    
    def get_chat_history(self):
        """Get the current chat history"""
        return self.chat_history
