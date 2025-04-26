from flask import Flask, request
from flask_cors import CORS
from flask_jwt_extended import JWTManager
import os
import requests
import logging
import subprocess
import webbrowser
import threading

from routes import init_routes
from config import (
    FRONTEND_NGROK_URL, FRONTEND_LOCAL_URL,
    BACKEND_NGROK_URL, BACKEND_LOCAL_URL,
    LOCAL_IP, BACKEND_PORT, LLM_MODEL, OLLAMA_URL,
    ALLOWED_ORIGINS
)

app = Flask(__name__)

# JWT configuration
app.config['JWT_SECRET_KEY'] = 'super-secret-dev-key'
app.config["JWT_TOKEN_LOCATION"] = ["headers"]
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = False  # Optional: never expires, or use timedelta
jwt = JWTManager(app)

# CORS setup (for JWT in headers, no cookies)
CORS(app, resources={
    r"/api/*": {
        "origins": ALLOWED_ORIGINS,
        "supports_credentials": False  # Cookies not needed with JWT
    }
})


# Load LLM model on startup
def load_model():
    print(f"Loading model {LLM_MODEL} into memory...")
    try:
        payload = {
            "model": LLM_MODEL,
            "prompt": "Model loading test",
            "stream": False
        }
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        print(f"Model {LLM_MODEL} loaded successfully.")
    except Exception as e:
        print(f"Failed to load model {LLM_MODEL}: {str(e)}")
        raise Exception("Model loading failed. Ensure Ollama is running and the model is available.")

# Initialize routes (they now use JWT)
init_routes(app)
print("🚀 Routes initialized")
app.logger.setLevel(logging.DEBUG)

# Load model
with app.app_context():
    load_model()


def open_browser():
    webbrowser.open_new("http://127.0.0.1:5173")


if __name__ == '__main__':
    print(f"Backend running with IP: {LOCAL_IP}, Port: {BACKEND_PORT}")
    print(f"Frontend Ngrok URL: {FRONTEND_NGROK_URL}")
    print(f"Backend Ngrok URL: {BACKEND_NGROK_URL}")
    print(f"Frontend Local URL: {FRONTEND_LOCAL_URL}")
    print(f"Backend Local URL: {BACKEND_LOCAL_URL}")
    print("Opening browser...")
    threading.Timer(1.0, open_browser).start()  # Open browser after 1 sec
    app.run(host='0.0.0.0', port=BACKEND_PORT, debug=False)


# npm run dev -- --host 127.0.0.1
# python App.py