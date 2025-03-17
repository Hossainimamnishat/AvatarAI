from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import dlib
import os
import shutil
import onnxruntime as ort
import trimesh
from PIL import Image
import mediapipe as mp
import uvicorn

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
MODEL_DIR = "models/"
OUTPUT_DIR = "generated_models/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount static files for serving 3D models
app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")

# Load Dlib face predictor (fallback)
FACE_MODEL_PATH = os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat")
if not os.path.exists(FACE_MODEL_PATH):
    raise FileNotFoundError("Face landmark model not found. Place 'shape_predictor_68_face_landmarks.dat' in 'models/'")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(FACE_MODEL_PATH)

# Load ONNX gender classification model
GENDER_MODEL_PATH = os.path.join(MODEL_DIR, "gender_model.onnx")
if not os.path.exists(GENDER_MODEL_PATH):
    raise FileNotFoundError("Gender classification model not found. Place 'gender_model.onnx' in 'models/'")

session = ort.InferenceSession(GENDER_MODEL_PATH)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

def detect_gender(image):
    """Predict gender using ONNX deep learning model"""
    try:
        resized = cv2.resize(image, (224, 224))
        blob = resized.astype(np.float32) / 255.0
        blob = np.expand_dims(blob.transpose(2, 0, 1), axis=0)

        inputs = {session.get_inputs()[0].name: blob}
        output = session.run(None, inputs)[0]

        return "male" if output[0][0] > output[0][1] else "female"
    except Exception as e:
        print(f"Gender detection error: {e}")
        return "unknown"

def detect_facial_features(image):
    """Detect facial landmarks using MediaPipe Face Mesh"""
    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            print("⚠️ No face detected")
            return None
        return results.multi_face_landmarks[0]  # Return the first face's landmarks
    except Exception as e:
        print(f"Facial feature detection error: {e}")
        return None

def extract_face_texture(image, landmarks):
    """Extract facial texture using convex hull"""
    try:
        # Convert landmarks to numpy array
        points = np.array([[int(lm.x * image.shape[1]), int(lm.y * image.shape[0])] for lm in landmarks.landmark])
        hull = cv2.convexHull(points)
        mask = np.zeros_like(image)
        cv2.fillConvexPoly(mask, hull, (255, 255, 255))
        face_region = cv2.bitwise_and(image, mask)

        # Crop the face region
        x, y, w, h = cv2.boundingRect(hull)
        face_region = face_region[y:y+h, x:x+w]

        if face_region.size == 0:
            return None

        texture_path = os.path.join(OUTPUT_DIR, "face_texture.jpg")
        Image.fromarray(cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)).save(texture_path)
        return texture_path
    except Exception as e:
        print(f"Texture extraction error: {e}")
        return None

def extract_skin_tone(image, landmarks):
    """Extract average skin tone from the face region"""
    try:
        points = np.array([[int(lm.x * image.shape[1]), int(lm.y * image.shape[0])] for lm in landmarks.landmark])
        hull = cv2.convexHull(points)
        mask = np.zeros_like(image)
        cv2.fillConvexPoly(mask, hull, (255, 255, 255))
        face_region = cv2.bitwise_and(image, mask)

        # Calculate average skin tone
        skin_tone = cv2.mean(face_region, mask=mask[:, :, 0])
        return skin_tone[:3]  # Return RGB values
    except Exception as e:
        print(f"Skin tone extraction error: {e}")
        return None

def apply_skin_tone(model, skin_tone):
    """Apply skin tone to the 3D model"""
    try:
        # Adjust the texture or material of the 3D model to match the skin tone
        model.visual.material.baseColorFactor = list(skin_tone) + [1.0]  # Add alpha channel
        return model
    except Exception as e:
        print(f" Skin tone application error: {e}")
        return model

def modify_3d_model(model, landmarks):
    """Modify 3D model to match detected face shape"""
    try:
        vertices = model.vertices.copy()

        # Example: Adjust jawline based on landmarks
        jaw_points = np.array([[int(lm.x * 1000), int(lm.y * 1000)] for lm in landmarks.landmark[0:17]])
        for i in range(len(jaw_points)):
            vertices[i, 0] += jaw_points[i][0] * 0.001
            vertices[i, 1] += jaw_points[i][1] * 0.001

        model.vertices = vertices
        return model
    except Exception as e:
        print(f"Error modifying 3D model: {e}")
        return model

def generate_3d_model(gender, landmarks):
    """Generate a realistic 3D avatar with facial features applied"""
    try:
        base_model_path = os.path.join(MODEL_DIR, f"{gender}_avatar.glb")
        if not os.path.exists(base_model_path):
            print(f" 3D model not found: {base_model_path}")
            return None

        output_path = os.path.join(OUTPUT_DIR, "avatar.glb")
        shutil.copy(base_model_path, output_path)

        # Load model & modify
        model = trimesh.load(output_path, force='mesh')  # Ensure it's a mesh
        model = modify_3d_model(model, landmarks)  # Adjust face

        # Extract & Apply texture
        texture_path = extract_face_texture(cv2.imread(base_model_path), landmarks)
        if texture_path:
            model = apply_texture_to_model(model, texture_path)

        model.export(output_path)

        return "avatar.glb"
    except Exception as e:
        print(f"3D model generation error: {e}")
        return None

@app.post("/upload/")
async def upload_images(front: UploadFile):
    """Handles image upload, detects gender, extracts facial features, and generates a 3D avatar"""
    try:
        image_bytes = await front.read()
        np_image = np.array(bytearray(image_bytes), dtype=np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        if image is None:
            print(" Image decoding failed")
            return {"error": "Invalid image"}

        print(" Image received, starting detection...")

        gender = detect_gender(image)
        print(f" Detected Gender: {gender}")

        landmarks = detect_facial_features(image)
        if not landmarks:
            print(" No facial landmarks detected")
            return {"error": "No face detected"}

        # Extract skin tone
        skin_tone = extract_skin_tone(image, landmarks)
        if skin_tone:
            print(f" Extracted Skin Tone: {skin_tone}")

        model_name = generate_3d_model(gender, landmarks)
        if not model_name:
            print(" 3D model generation failed")
            return {"error": "3D model generation failed"}

        # Apply skin tone to the 3D model
        model = trimesh.load(os.path.join(OUTPUT_DIR, model_name), force='mesh')
        model = apply_skin_tone(model, skin_tone)
        model.export(os.path.join(OUTPUT_DIR, model_name))

        print(" Avatar generated successfully!")

        return {
            "gender": gender,
            "avatar_url": f"http://localhost:8000/static/{model_name}"
        }

    except Exception as e:
        print(f" Server error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)