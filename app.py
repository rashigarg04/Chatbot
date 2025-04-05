from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import time
import os
from functools import wraps
import threading
import queue
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Read API key from environment variable for better security
API_KEY = os.environ.get("AIzaSyB38YsqlhJZZg6IFbP-liuMiTTVG06kf0M", "AIzaSyAX9DVo1ADQBARl5xtE-uUudXbOTbHVQl8")
genai.configure(api_key=API_KEY)

# Enhanced generation configuration
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "stop_sequences": ["bye", "exit", "quit", "goodbye"],
    "response_mime_type": "text/plain",
}

# Safety settings
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
]

# Career guidance system instructions
career_guidance_instructions = """
According to NCRB data, states like Rajasthan, Uttar Pradesh, Madhya Pradesh, and Maharashtra report the highest number of rape cases in India, with Rajasthan alone registering over 5,000 cases in 2022. Alarming trends also show that a large number of victims are minors, especially in states like Goa and Uttarakhand. To stay safe, women should avoid dark or isolated areas, always share live location with trusted contacts, carry personal safety tools like pepper spray, avoid using headphones while walking alone at night, and stay alert to their surroundings. Your AI-based safety system can use real-time location and crime data to mark high-risk zones, send alerts when users enter unsafe areas, offer personalized safety tips, allow anonymous incident reporting, and include an SOS button to notify emergency contacts and authorities instantly.


"""

# Initialize Flask app
app = Flask(__name__)

# Rate limiting implementation
class RateLimiter:
    def __init__(self, max_calls_per_minute=60):
        self.max_calls = max_calls_per_minute
        self.calls = queue.Queue()
        self.lock = threading.Lock()
    
    def __call__(self, f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            with self.lock:
                now = time.time()
                
                # Remove calls older than 1 minute
                while not self.calls.empty() and now - self.calls.queue[0] > 60:
                    self.calls.get()
                
                # Check if we're at the limit
                if self.calls.qsize() >= self.max_calls:
                    return jsonify({
                        "response": "Our service is experiencing high demand. Please try again in a few moments.",
                        "error": "rate_limit_exceeded"
                    }), 429
                
                # Add this call to the queue
                self.calls.put(now)
            
            return f(*args, **kwargs)
        return wrapped

# Create rate limiter with a conservative limit (adjust based on your quota)
rate_limiter = RateLimiter(max_calls_per_minute=50)

# Initialize model with a function to handle retries and model switching
def get_model():
    # Try primary model first
    try:
        return genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            safety_settings=safety_settings,
            system_instruction=career_guidance_instructions
        )
    except Exception as e:
        logger.warning(f"Failed to initialize primary model: {e}")
        # Fallback to a different model
        return genai.GenerativeModel(
            model_name="gemini-1.0-pro",  # Fallback model
            generation_config=generation_config,
            safety_settings=safety_settings,
            system_instruction=career_guidance_instructions
        )

# Route for the chatbot interface
@app.route("/")
def index():
    return render_template("index.html")

# Route for chatbot interaction with rate limiting
@app.route("/chat", methods=["POST"])
@rate_limiter
def chat():
    data = request.json
    user_input = data.get("message", "")
    history = data.get("history", [])
    
    # Convert history to the format expected by Gemini API
    formatted_history = []
    if history:
        for message in history:
            role = message.get("role")
            content = message.get("parts", [""])[0]
            if role and content:
                formatted_history.append({"role": role, "parts": [content]})
    
    # Maximum retry attempts
    max_retries = 3
    retry_count = 0
    backoff_time = 1  # Initial backoff time in seconds
    
    while retry_count < max_retries:
        try:
            # Get model (will try primary first, then fallback)
            model = get_model()
            
            # Start chat session with properly formatted history
            chat_session = model.start_chat(history=formatted_history)
            
            # Send message and get response
            response = chat_session.send_message(user_input)
            model_response = response.text
            
            # Append to history
            updated_history = history.copy() if history else []
            updated_history.append({"role": "user", "parts": [user_input]})
            updated_history.append({"role": "model", "parts": [model_response]})
            
            return jsonify({"response": model_response, "history": updated_history})
        
        except Exception as e:
            error_message = str(e)
            logger.error(f"API Error (attempt {retry_count+1}/{max_retries}): {error_message}")
            
            # Check if it's a quota or rate limit error
            if "RATE_LIMIT_EXCEEDED" in error_message or "Quota exceeded" in error_message:
                retry_count += 1
                if retry_count < max_retries:
                    logger.info(f"Rate limit hit, backing off for {backoff_time} seconds")
                    time.sleep(backoff_time)
                    backoff_time *= 2  # Exponential backoff
                    continue
                
                # If we've exhausted retries, return a user-friendly message
                friendly_message = (
                    "I'm sorry, but our service is currently experiencing high demand. "
                    "Please try again in a few minutes."
                )
            else:
                # For other errors, don't retry
                friendly_message = (
                    "I apologize, but I encountered an error processing your request. "
                    "Please try again with a different question."
                )
                break
    
    # Add error to history so the conversation can continue
    updated_history = history.copy() if history else []
    updated_history.append({"role": "user", "parts": [user_input]})
    updated_history.append({"role": "model", "parts": [friendly_message]})
    
    return jsonify({"response": friendly_message, "history": updated_history})

# Add health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "api_status": "connected" if API_KEY else "missing_key"}), 200

# Add endpoint to check current rate limit status
@app.route("/status", methods=["GET"])
def rate_status():
    with rate_limiter.lock:
        now = time.time()
        
        # Remove calls older than 1 minute
        while not rate_limiter.calls.empty() and now - rate_limiter.calls.queue[0] > 60:
            rate_limiter.calls.get()
        
        current_usage = rate_limiter.calls.qsize()
        capacity = rate_limiter.max_calls
        
    return jsonify({
        "current_usage": current_usage,
        "capacity": capacity,
        "available": capacity - current_usage,
        "status": "healthy" if current_usage < capacity else "at_capacity"
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
