"""
FastAPI application for meme political classification.

This API allows users to upload meme images and get predictions on whether
the meme is political or non-political.

Pipeline:
1. Upload meme image
2. Extract text using Gemini API
3. Clean and normalize extracted text
4. Predict using PIXAI model (image + text)
5. Return prediction with confidence score
"""

import os
from io import BytesIO
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict
from PIL import Image
import torch

from gemini_extractor import GeminiTextExtractor
from text_normalizer import process_extracted_text
from model import load_model, predict


# ===== Configuration =====
class Config:
    """Application configuration."""

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    MODEL_PATH = os.getenv("MODEL_PATH", "model_task1.pth")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_LEN = 64
    NUM_HEADS = 8


# ===== Response Models =====
class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""

    success: bool
    prediction: str
    confidence: float
    political_probability: float
    non_political_probability: float
    extracted_text: list
    normalized_text: str
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check."""

    model_config = ConfigDict(protected_namespaces=())

    status: str
    device: str
    model_loaded: bool


# ===== Global Variables =====
config = Config()
gemini_extractor = None
pixai_model = None
tokenizer = None
clip_preprocess = None


# ===== Lifespan Context Manager =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models and services on startup."""
    global gemini_extractor, pixai_model, tokenizer, clip_preprocess

    # Check for Gemini API key
    if not config.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable not set. "
            "Please set it before starting the server."
        )

    # Check for model file
    if not os.path.exists(config.MODEL_PATH):
        raise RuntimeError(
            f"Model file not found at {config.MODEL_PATH}. "
            f"Please set MODEL_PATH environment variable to the correct path."
        )

    print("Initializing Gemini text extractor...")
    gemini_extractor = GeminiTextExtractor(api_key=config.GEMINI_API_KEY)

    print(f"Loading PIXAI model from {config.MODEL_PATH}...")
    print(f"Using device: {config.DEVICE}")
    pixai_model, tokenizer, clip_preprocess = load_model(
        model_path=config.MODEL_PATH,
        device=config.DEVICE,
        max_len=config.MAX_LEN,
        num_heads=config.NUM_HEADS,
    )

    print("✓ All models loaded successfully!")
    yield
    # Cleanup on shutdown (if needed)
    print("Shutting down...")


# ===== Initialize FastAPI App =====
app = FastAPI(
    title="Meme Political Classifier API",
    description="API for classifying memes as political or non-political using vision-language models",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== Health Check Endpoint =====
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API is running and models are loaded."""
    return HealthResponse(
        status="healthy", device=config.DEVICE, model_loaded=pixai_model is not None
    )


# ===== Main Prediction Endpoint =====
@app.post("/predict", response_model=PredictionResponse)
async def predict_meme(file: UploadFile = File(...)):
    """
    Upload a meme image and get a prediction.

    Args:
        file: Uploaded image file (JPEG, PNG, etc.)

    Returns:
        Prediction result with confidence score
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, detail="Uploaded file must be an image"
            )

        # Read and open image
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")

        # Step 1: Extract text using Gemini
        print("Extracting text with Gemini...")
        extracted_text = gemini_extractor.extract_text(image)

        if not extracted_text:
            # If no text extracted, use empty string
            print("Warning: No text extracted from image")
            extracted_text = []

        # Step 2: Clean and normalize text
        print("Cleaning and normalizing text...")
        normalized_text = process_extracted_text(extracted_text)

        # If normalized text is empty, use a placeholder
        if not normalized_text.strip():
            normalized_text = "[no text detected]"

        # Step 3: Make prediction
        print("Making prediction...")
        result = predict(
            model=pixai_model,
            tokenizer=tokenizer,
            preprocess=clip_preprocess,
            image=image,
            text=normalized_text,
            device=config.DEVICE,
            max_len=config.MAX_LEN,
        )

        # Return response
        return PredictionResponse(
            success=True,
            prediction=result["prediction"],
            confidence=result["confidence"],
            political_probability=result["political_probability"],
            non_political_probability=result["non_political_probability"],
            extracted_text=extracted_text,
            normalized_text=normalized_text,
            message="Prediction completed successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# ===== Root Endpoint =====
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web interface."""
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return """
        <html>
            <body>
                <h1>Meme Political Classifier API</h1>
                <p>Web interface not found. Use the API endpoints:</p>
                <ul>
                    <li>POST /predict - Upload image and get prediction</li>
                    <li>GET /health - Check API health status</li>
                    <li>GET /docs - Interactive API documentation</li>
                </ul>
            </body>
        </html>
        """


@app.get("/api")
async def api_info():
    """API information endpoint."""
    return {
        "name": "Meme Political Classifier API",
        "version": "1.0.0",
        "description": "Upload meme images to classify as political or non-political",
        "endpoints": {
            "POST /predict": "Upload image and get prediction",
            "GET /health": "Check API health status",
            "GET /docs": "Interactive API documentation",
        },
    }


if __name__ == "__main__":
    import uvicorn

    # Run the server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
