"""
Simple test client for the Meme Political Classifier API.

Usage:
    python test_api.py path/to/meme.jpg
"""

import sys
import requests
from pathlib import Path


def test_predict(image_path: str, api_url: str = "http://localhost:8000"):
    """
    Test the /predict endpoint with an image.

    Args:
        image_path: Path to the image file
        api_url: Base URL of the API
    """
    # Check if file exists
    if not Path(image_path).exists():
        print(f"Error: File not found: {image_path}")
        return

    # Test health endpoint first
    print(f"Testing API health at {api_url}/health...")
    try:
        health_response = requests.get(f"{api_url}/health")
        health_response.raise_for_status()
        health_data = health_response.json()
        print(f"✓ API Status: {health_data['status']}")
        print(f"✓ Device: {health_data['device']}")
        print(f"✓ Model Loaded: {health_data['model_loaded']}")
        print()
    except requests.exceptions.RequestException as e:
        print(f"✗ Error connecting to API: {e}")
        print("Make sure the API server is running on http://localhost:8000")
        return

    # Test prediction endpoint
    print(f"Uploading image: {image_path}")
    print("Waiting for prediction...")

    try:
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = requests.post(f"{api_url}/predict", files=files)
            response.raise_for_status()

        result = response.json()

        # Print results
        print("\n" + "=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        print(f"Success: {result['success']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nProbabilities:")
        print(f"  Political: {result['political_probability']:.2%}")
        print(f"  Non-Political: {result['non_political_probability']:.2%}")
        print(f"\nExtracted Text Blocks ({len(result['extracted_text'])} found):")
        for i, text in enumerate(result["extracted_text"], 1):
            print(f"  {i}. {text}")
        print(f"\nNormalized Text:")
        print(f"  {result['normalized_text']}")
        print("=" * 60)

    except requests.exceptions.RequestException as e:
        print(f"\n✗ Error during prediction: {e}")
        if hasattr(e.response, "text"):
            print(f"Response: {e.response.text}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <path_to_image>")
        print("\nExample:")
        print("  python test_api.py meme.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    test_predict(image_path)
