from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from skimage import transform
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from google.cloud import storage

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure GCS
storage_client = storage.Client()
bucket_name = "saved-image"
bucket = storage_client.bucket(bucket_name)

MODELS = {
    "bellpepper": {
        "path": "fastapi-image-recognition-vegetable-ripness/saved-models/BellPaper_DenseNet2_model",
        "class_names": ["ripe", "old", "damaged", "dried", "unripe"]
    },
    "chilepepper": {
        "path": "fastapi-image-recognition-vegetable-ripness/saved-models/ChilePaper_DenseNet2_model",
        "class_names": ["old", "unripe", "dried", "ripe", "damaged"]
    },
    "tomato": {
        "path": "fastapi-image-recognition-vegetable-ripness/saved-models/Tomato_DenseNet2_model",
        "class_names": ["damaged", "old", "ripe", "unripe"]
    }
}

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

def preprocess_image(image):
    image = transform.resize(image, (224, 224, 3))
    image = np.expand_dims(image, 0)
    return image

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

@app.on_event("startup")
async def startup_event():
    app.models = {}
    for model_name, model_config in MODELS.items():
        model = load_model(model_config["path"])
        app.models[model_name] = model

def upload_file_to_gcs(file, filename):
    blob = bucket.blob(filename)
    blob.upload_from_file(file, content_type=file.content_type)

@app.post("/upload-image")
async def UploadImage(
    file: UploadFile = File(...)
):
    if file.content_type.startswith('image/'):
        model_names = MODELS.keys()
        model_predictions = {}

        for model_name in model_names:
            if model_name in app.models:
                model = app.models[model_name]

                image = read_file_as_image(await file.read())
                preprocessed_image = preprocess_image(image)
                predictions = model.predict(preprocessed_image)

                class_names = MODELS[model_name]["class_names"]
                predicted_class = class_names[np.argmax(predictions[0])]
                confidence = np.max(predictions[0])

                model_predictions[model_name] = {
                    "class": predicted_class,
                    "confidence": float(confidence)
                }

        # Upload image to Google Cloud Storage
        filename = file.filename
        upload_file_to_gcs(file.file, filename)

        return {
            "predictions": model_predictions,
            "filename": filename
        }
    else:
        return {"error": "Invalid file format"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)