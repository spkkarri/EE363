from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import time
from PIL import Image
from models.model_loader import SegmentationModel

# Initialize Flask app
app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
UPLOAD_DIR = os.path.join(STATIC_DIR, 'uploads')
RESULT_DIR = os.path.join(STATIC_DIR, 'results')

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Model paths
MODEL_PATH = os.path.join(BASE_DIR, "DG_segformerB3_model.pth")
CLASS_CSV = os.path.join(BASE_DIR, "class_dict.csv")

# Initialize model
print("Loading segmentation model...")
start_time = time.time()
model = SegmentationModel(MODEL_PATH, CLASS_CSV)
print(f"Model loaded in {time.time()-start_time:.2f} seconds")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def handle_upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if not allowed_file(file.filename):
        return jsonify({"error": "Allowed file types: png, jpg, jpeg"}), 400

    try:
        # Save with unique filename
        timestamp = int(time.time())
        upload_filename = f"upload_{timestamp}.jpg"
        upload_path = os.path.join(UPLOAD_DIR, upload_filename)
        file.save(upload_path)
        
        # Process image
        start_time = time.time()
        original, mask = model.predict(upload_path)
        processing_time = time.time() - start_time
        
        # Save results
        original_filename = f"original_{timestamp}.jpg"
        mask_filename = f"mask_{timestamp}.jpg"
        
        original_path = os.path.join(RESULT_DIR, original_filename)
        mask_path = os.path.join(RESULT_DIR, mask_filename)
        
        Image.fromarray(original).save(original_path)
        Image.fromarray(mask).save(mask_path)
        
        return jsonify({
            "original": f"/results/{original_filename}",
            "mask": f"/results/{mask_filename}",
            "time": f"{processing_time:.2f}s"
        })
        
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500
        
    finally:
        if 'upload_path' in locals() and os.path.exists(upload_path):
            os.remove(upload_path)

@app.route('/results/<filename>')
def serve_result(filename):
    return send_from_directory(RESULT_DIR, filename)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)

if __name__ == '__main__':
    port = 5001
    print(f"\nServer running at http://localhost:{port}")
    print("Press Ctrl+C to stop\n")
    app.run(host='0.0.0.0', port=port, threaded=True)