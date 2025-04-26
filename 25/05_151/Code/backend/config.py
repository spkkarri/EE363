import platform
import subprocess

# Detect and set local IP
if platform.system() == "Windows":
    ipconfig_output = subprocess.run(["ipconfig"], capture_output=True, text=True).stdout
    for line in ipconfig_output.splitlines():
        if "IPv4" in line:
            LOCAL_IP = line.split(":")[-1].strip()
            break
else:
    LOCAL_IP = subprocess.run(['hostname', '-I'], capture_output=True, text=True, check=True).stdout.split()[0]

# For local development, override auto-detected IP with localhost
LOCAL_IP = "127.0.0.1"  # Replace with your local IP if needed

# Frontend/Backend URLs (default to localhost)
FRONTEND_LOCAL_URL = f"http://{LOCAL_IP}:5173"
BACKEND_LOCAL_URL = f"http://{LOCAL_IP}:5000"

# Optional: Use ngrok URLs in production (commented out here)
# FRONTEND_NGROK_URL = "https://your-frontend-ngrok-url.ngrok-free.app"
# BACKEND_NGROK_URL = "https://your-backend-ngrok-url.ngrok-free.app"

# Use local URLs for both dev and testing
FRONTEND_NGROK_URL = FRONTEND_LOCAL_URL
BACKEND_NGROK_URL = BACKEND_LOCAL_URL

# Allowed CORS origins
ALLOWED_ORIGINS = [FRONTEND_LOCAL_URL]

# Backend port
BACKEND_PORT = 5000

# LLM model + Ollama config
LLM_MODEL = "deepseek-coder"
OLLAMA_URL = "http://localhost:11434/api/generate"
