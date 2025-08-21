import traceback
import time
from flask import Flask, request, jsonify
from deepface import DeepFace
import cv2
import numpy as np
import base64
from flask_cors import CORS

# ====================
# Initialize Flask App
# ====================
app = Flask(__name__)
CORS(app, resources={
    r"/generate-embedding": {"origins": "http://localhost:5173"},
    r"/compare-embeddings": {"origins": "http://localhost:5173"}
})

# ====================
# Preload Facenet Model
# ====================
print("🔧 Loading SFace model at startup...")
facenet_model = DeepFace.build_model("SFace")
print("✅ SFace model loaded.")

# ====================
# Utility Functions
# ====================
def decode_image(image_data):
    try:
        print("🧠 Decoding base64 image...")
        image_data = image_data.split(',')[1]  # remove base64 prefix
        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        print("✅ Image decoded successfully.")
        return img
    except Exception as e:
        print("❌ Decoding error:", e)
        return None

def is_real_face(image, threshold=5):
    try:
        print("🔎 Checking for real face (anti-spoofing)...")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        print(f"📏 Laplacian Variance: {laplacian_var}")
        return laplacian_var > threshold
    except Exception as e:
        print("❌ Error in real face check:", e)
        return False

# =============================
# 1. Generate Embedding Endpoint
# =============================
@app.route('/generate-embedding', methods=['POST', 'OPTIONS'])
def generate_embedding():
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()

    try:
        print("🔔 Received request for /generate-embedding")
        start_time = time.time()

        data = request.get_json()
        if 'image' not in data:
            return jsonify({"status": "error", "message": "No image provided"}), 400

        img = decode_image(data['image'])
        if img is None:
            return jsonify({"status": "error", "message": "Invalid image"}), 400

        if not is_real_face(img):
            return jsonify({"status": "error", "message": "Fake or spoofed face detected"}), 403

        model_start = time.time()
        print("📡 Generating embedding... (SFace default loading)")
        embedding_result = DeepFace.represent(
            img_path=img,
            model_name="SFace",
            enforce_detection=False  # ✅ prevents crash if no face
        )

        if not embedding_result or "embedding" not in embedding_result[0]:
            print("❌ No face detected or embedding missing.")
            return _corsify_actual_response(jsonify({
                "status": "error",
                "message": "No face detected in the image"
            })), 400

        embedding = embedding_result[0]["embedding"]
        model_end = time.time()

        print(f"✅ Embedding generated in {(model_end - model_start):.2f} sec")
        print(f"⏱️ Total /generate-embedding time: {(time.time() - start_time):.2f} sec")

        return _corsify_actual_response(jsonify({
            "status": "success",
            "embedding": embedding
        }))

    except Exception as e:
        print("❌ Error in generate-embedding:", e)
        print(traceback.format_exc())
        return _corsify_actual_response(jsonify({
            "status": "error",
            "message": str(e)
        })), 500

# =============================
# 2. Compare Embeddings Endpoint
# =============================
@app.route('/compare-embeddings', methods=['POST'])
def compare_embeddings():
    try:
        print("🔔 Received request for /compare-embeddings")
        start_time = time.time()

        data = request.get_json()
        if not data or 'embedding1' not in data or 'embedding2' not in data:
            return jsonify({"status": "error", "message": "Both embeddings are required"}), 400

        try:
            emb1 = np.array(data['embedding1'], dtype=np.float32)
            emb2 = np.array(data['embedding2'], dtype=np.float32)
        except Exception as e:
            return jsonify({"status": "error", "message": f"Invalid embedding format: {str(e)}"}), 400

        if emb1.shape != emb2.shape:
            return jsonify({"status": "error", "message": f"Embedding shape mismatch: {emb1.shape} vs {emb2.shape}"}), 400

        distance = float(np.linalg.norm(emb1 - emb2))
        threshold = 10.0
        is_match = bool(distance < threshold)

        print(f"✅ Comparison done. Distance: {distance:.4f}, Match: {is_match}, Time: {(time.time() - start_time):.2f} sec")

        return jsonify({
            "status": "success",
            "verified": is_match,
            "distance": distance,
            "threshold": threshold
        })

    except Exception as e:
        print("❌ Error in compare-embeddings:", e)
        print(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

# =============================
# CORS Helpers
# =============================
def _build_cors_preflight_response():
    response = jsonify({"status": "success"})
    response.headers.add("Access-Control-Allow-Origin", "http://localhost:5173")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "http://localhost:5173")
    return response

# =============================
# Start Flask Server
# =============================
if __name__ == '__main__':
    print("🚀 Flask server running on port 5005...")
    app.run(port=5005)
