from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
from fastapi.staticfiles import StaticFiles  # To serve audio files
import os
from google.oauth2 import service_account
from google.cloud import vision, texttospeech
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import base64
from PIL import Image

app = FastAPI()

# Set up CORS middleware for React Native
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace '*' with specific origins like 'http://localhost:19000' if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up directories
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
YOLO_MODEL_PATH = BASE_DIR / "best.pt"
SERVICE_ACCOUNT_KEY = BASE_DIR / "text-detection-project-440901-30657f270abf.json"

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Set up Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(SERVICE_ACCOUNT_KEY)

# Initialize Google Vision API and Text-to-Speech clients
vision_client = vision.ImageAnnotatorClient()
tts_client = texttospeech.TextToSpeechClient()

# Load YOLO model
yolo_model = YOLO(str(YOLO_MODEL_PATH))

# Serve the output directory to expose audio files for React Native app
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")


@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    """Process the uploaded image, detect text, and generate audio."""
    # Validate file type
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file type. Only image files are allowed.")

    # Save uploaded image to `uploads/`
    uploaded_file_path = UPLOAD_DIR / file.filename
    try:
        with open(uploaded_file_path, "wb") as buffer:
            buffer.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Load image with OpenCV
    image = cv2.imread(str(uploaded_file_path))
    if image is None:
        raise HTTPException(status_code=500, detail="Failed to read the image file.")

    # YOLO detection
    try:
        results = yolo_model(image, conf=0.5)  # Lowering the confidence threshold to 0.5
        print("YOLO Results: ", results)  # Debugging: Check YOLO results

        bounding_boxes = results[0].boxes.xyxy.tolist()

        # If bounding boxes are detected, crop the image based on the boxes
        if bounding_boxes:
            merged_boxes = merge_boxes(bounding_boxes)
            # Crop the image using the merged bounding boxes
            cropped_image = crop_image(image, merged_boxes)
        else:
            # No bounding boxes detected, use the entire image
            cropped_image = image

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in YOLO detection: {str(e)}")

    # Perform OCR and Text-to-Speech
    detected_text = await perform_ocr(cropped_image)
    if not detected_text.strip():
        raise HTTPException(status_code=404, detail="No text detected.")

    audio_file_path = await perform_text_to_speech(detected_text)

    return {
        "text": detected_text,
        "audio_file": f"/output/{audio_file_path.name}",  # Accessible URL for the audio file
        "image_preview": encode_image_as_base64(cropped_image)
    }


def merge_boxes(boxes, threshold=10):
    """Merge overlapping or close bounding boxes."""
    merged_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        merged = False
        for merged_box in merged_boxes:
            mx_min, my_min, mx_max, my_max = map(int, merged_box)
            if (x_min < mx_max + threshold and x_max > mx_min - threshold and
                y_min < my_max + threshold and y_max > my_min - threshold):
                new_box = [
                    min(x_min, mx_min),
                    min(y_min, my_min),
                    max(x_max, mx_max),
                    max(y_max, my_max)
                ]
                merged_boxes.remove(merged_box)
                merged_boxes.append(new_box)
                merged = True
                break
        if not merged:
            merged_boxes.append(box)
    return merged_boxes


def crop_image(image, bounding_boxes):
    """Crop the image based on the given bounding boxes."""
    # Find the bounding box that covers all the detected text regions
    x_min_all = min(box[0] for box in bounding_boxes)
    y_min_all = min(box[1] for box in bounding_boxes)
    x_max_all = max(box[2] for box in bounding_boxes)
    y_max_all = max(box[3] for box in bounding_boxes)

    cropped_image = image[int(y_min_all):int(y_max_all), int(x_min_all):int(x_max_all)]
    return cropped_image


async def perform_ocr(image):
    """Perform OCR using Google Vision API."""
    # Save the image for OCR processing
    cropped_image_path = OUTPUT_DIR / "cropped_text_region.jpg"
    cv2.imwrite(str(cropped_image_path), image)

    # Vision API OCR
    try:
        with open(cropped_image_path, "rb") as image_file:
            content = image_file.read()
        vision_image = vision.Image(content=content)
        response = vision_client.text_detection(image=vision_image)
        texts = response.text_annotations
        return texts[0].description if texts else ""
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during OCR: {str(e)}")


async def perform_text_to_speech(text):
    """Convert text to speech using Google Text-to-Speech API."""
    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

        # Generate speech
        response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        audio_file_path = OUTPUT_DIR / "detected_text_audio.mp3"
        with open(audio_file_path, "wb") as out:
            out.write(response.audio_content)
        return audio_file_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during Text-to-Speech: {str(e)}")


def encode_image_as_base64(image):
    """Convert image to a base64-encoded string for preview."""
    _, img_bytes = cv2.imencode('.jpg', image)
    img_str = base64.b64encode(img_bytes).decode('utf-8')
    return img_str


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
