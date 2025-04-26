# run_all.py
import subprocess
import time
import os


backend_path = os.path.join("backend", "App.py")  # Correct file name and location
if not os.path.exists(backend_path):
    raise FileNotFoundError(f"Backend file not found: {backend_path}")

frontend_path = os.path.join("frontend", "package.json")
if not os.path.exists(frontend_path):
    raise FileNotFoundError("Frontend package.json not found in ./frontend")

Ragpdf_path = os.path.join("RAG - pdf", "RAG_app.py")  # Correct file name and location
if not os.path.exists(Ragpdf_path):
    raise FileNotFoundError(f"RAG PDF file not found: {Ragpdf_path}")


# Start the Python backend
backend = subprocess.Popen(["python", backend_path])
ragpdf = subprocess.Popen(["python", Ragpdf_path])

# Wait a moment to ensure backend initializes
time.sleep(2)

# Start the frontend (assumes you have `npm run dev` or `yarn dev`)
frontend = subprocess.Popen(["npm", "run", "dev", "--", "--host", "127.0.0.1"], cwd="frontend", shell=True)  # adjust path if needed

try:
    print("Backend is running...")
    print("RAG PDF is running...")
    print("Frontend is running...")

except KeyboardInterrupt:
    print("Shutting down...")
    backend.terminate()
    ragpdf.terminate()
    frontend.terminate()
