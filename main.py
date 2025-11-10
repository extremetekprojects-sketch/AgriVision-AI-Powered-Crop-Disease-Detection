
import os
import io
import warnings
import numpy as np
from PIL import Image
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import joblib
from ultralytics import YOLO
import tensorflow as tf

warnings.filterwarnings("ignore")

# --------------------------------------------------------------
# 1. VERIFY REQUIRED FILES
# --------------------------------------------------------------
required = [
    "best.pt",                    # Strawberry YOLO
    "best_model.keras",           # Tomato CNN
    "cucumber_leaf_model.keras",  # Cucumber CNN
    "Decision_Tree.pkl",
    "scaler.pkl"
]
missing = [f for f in required if not os.path.exists(f)]
if missing:
    raise FileNotFoundError(
        f"Missing files: {', '.join(missing)}. "
        f"Place them in: {os.getcwd()}"
    )

# --------------------------------------------------------------
# 2. GLOBAL MODELS
# --------------------------------------------------------------
yolo_model = None
tomato_model = None
cucumber_model = None
dt_model = None
scaler = None

# ----- Class Names -----
TOMATO_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

CUCUMBER_CLASSES = sorted([
    "Anthracnose",
    "Bacterial Wilt",
    "Belly Rot",
    "Downy Mildew",
    "Fresh Cucumber",
    "Fresh Leaf",
    "Gummy Stem Blight",
    "Pythium Fruit Rot"
])

label_map = {0: "Healthy", 1: "Moderate Stress", 2: "High Stress"}

# --------------------------------------------------------------
# 3. LOAD MODELS (once at startup)
# --------------------------------------------------------------
async def load_models():
    global yolo_model, tomato_model, cucumber_model, dt_model, scaler

    # Decision-Tree + Scaler
    if dt_model is None or scaler is None:
        print("Loading DT + Scaler...")
        dt_model = joblib.load("Decision_Tree.pkl")
        scaler   = joblib.load("scaler.pkl")
        if not hasattr(dt_model, "monotonic_cst"):
            dt_model.__dict__["monotonic_cst"] = None
        print("DT + Scaler loaded!")

    # Strawberry YOLO
    if yolo_model is None:
        print("Loading Strawberry YOLO...")
        yolo_model = YOLO("best.pt")
        print("Strawberry YOLO loaded!")

    # Tomato CNN
    if tomato_model is None:
        print("Loading Tomato CNN...")
        tomato_model = tf.keras.models.load_model("best_model.keras")
        print("Tomato CNN loaded!")

    # Cucumber CNN
    if cucumber_model is None:
        print("Loading Cucumber CNN...")
        cucumber_model = tf.keras.models.load_model("cucumber_leaf_model.keras")
        print("Cucumber CNN loaded!")


# --------------------------------------------------------------
# 4. PREPROCESSING FUNCTIONS
# --------------------------------------------------------------
def preprocess_150x150(img: Image.Image) -> np.ndarray:
    """Used by both Tomato and Cucumber models (150x150, /255)"""
    img = img.resize((150, 150))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1,150,150,3)
    return arr


# --------------------------------------------------------------
# 5. SENSOR DATA MODEL
# --------------------------------------------------------------
class SensorData(BaseModel):
    Plant_ID: int = 1
    Soil_Moisture: float
    Ambient_Temperature: float
    Soil_Temperature: float
    Humidity: float
    Light_Intensity: float
    Soil_pH: float
    Nitrogen_Level: float
    Phosphorus_Level: float
    Potassium_Level: float
    Chlorophyll_Content: float
    Electrochemical_Signal: float


# --------------------------------------------------------------
# 6. FASTAPI APP
# --------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    await load_models()
    yield

app = FastAPI(
    title="Multi-Plant Disease Detector (Strawberry + Tomato + Cucumber)",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------
# 7. SENSOR ENDPOINT
# --------------------------------------------------------------
@app.post("/predict/health")
async def predict_health(data: SensorData):
    try:
        await load_models()
        features = np.array([[
            data.Plant_ID, data.Soil_Moisture, data.Ambient_Temperature,
            data.Soil_Temperature, data.Humidity, data.Light_Intensity,
            data.Soil_pH, data.Nitrogen_Level, data.Phosphorus_Level,
            data.Potassium_Level, data.Chlorophyll_Content,
            data.Electrochemical_Signal
        ]])
        scaled = scaler.transform(features)
        pred = int(dt_model.predict(scaled)[0])
        raw = dt_model.predict_proba(scaled)[0]
        proba = np.exp(raw) / np.sum(np.exp(raw))
        confidence = proba.max() * 100

        return {
            "plant_health_status": label_map.get(pred, "Unknown"),
            "confidence": f"{confidence:.2f}%",
            "prediction_code": pred,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------
# 8. IMAGE DETECTION – 3 PLANTS
# --------------------------------------------------------------
@app.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    plant_type: str = Query("strawberry", enum=["strawberry", "tomato", "cucumber"])
):
    try:
        await load_models()
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # ------------------- STRAWBERRY (YOLO) -------------------
        if plant_type == "strawberry":
            results = yolo_model(image, verbose=False)
            detections = []
            for r in results:
                boxes = getattr(r, "boxes", None)
                if boxes is not None:
                    for box in boxes:
                        detections.append({
                            "class": r.names[int(box.cls)],
                            "confidence": float(box.conf),
                            "bbox": box.xyxy.tolist()[0],
                        })
            return {
                "plant_type": "strawberry",
                "detections": detections,
                "total_detections": len(detections)
            }

        # ------------------- TOMATO (CNN) -------------------
        elif plant_type == "tomato":
            arr = preprocess_150x150(image)
            preds = tomato_model.predict(arr, verbose=0)[0]
            idx = int(np.argmax(preds))
            confidence = float(preds[idx]) * 100

            return {
                "plant_type": "tomato",
                "predicted_class": TOMATO_CLASSES[idx],
                "confidence": f"{confidence:.2f}%",
                "all_probabilities": {
                    TOMATO_CLASSES[i]: f"{float(preds[i])*100:.2f}%"
                    for i in range(len(TOMATO_CLASSES))
                }
            }

        # ------------------- CUCUMBER (CNN) -------------------
        else:  # plant_type == "cucumber"
            arr = preprocess_150x150(image)
            preds = cucumber_model.predict(arr, verbose=0)[0]
            idx = int(np.argmax(preds))
            confidence = float(preds[idx]) * 100

            return {
                "plant_type": "cucumber",
                "predicted_class": CUCUMBER_CLASSES[idx],
                "confidence": f"{confidence:.2f}%",
                "all_probabilities": {
                    CUCUMBER_CLASSES[i]: f"{float(preds[i])*100:.2f}%"
                    for i in range(len(CUCUMBER_CLASSES))
                }
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------
# 9. ROOT
# --------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "message": "Multi-Plant Disease API is running",
        "endpoints": [
            "POST /predict/health",
            "POST /detect/image?plant_type=strawberry",
            "POST /detect/image?plant_type=tomato",
            "POST /detect/image?plant_type=cucumber"
        ],
        "status": "healthy"
    }


# --------------------------------------------------------------
# 10. RUN
# --------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info", reload=True)