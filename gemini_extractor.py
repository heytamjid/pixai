"""
Text extraction service using Google Gemini API.
Adapts the VLM extraction prompt from the Qwen VLM notebook.
"""

import json
import re
import google.generativeai as genai
from PIL import Image
from typing import List, Dict


class GeminiTextExtractor:
    """Extract text from meme images using Google Gemini API."""

    # VLM prompt adapted from Qwen VLM notebook
    EXTRACTION_PROMPT = """
Task: Perform a granular OCR scan of this image.
You are given a meme. You will extract EACH AND EVERY BLOCK of the VISIBLE TEXTS in this image.

Rules:
1. Scan the image from Top-Left to Bottom-Right.
2. Identify EVERY SEPARATE BLOCK of text.
3. SEPARATION RULE: If two pieces of text are visually separated, treat them as DIFFERENT list items. Do not merge them.
4. **CRITICAL:** Memes can have Bangla and/or English words mixed. If the text is Bangla, write in BANGLA SCRIPT. If English, write in English. Do NOT transcript to the other language.
5. Transcribe the text EXACTLY as written.
6. Do NOT output duplicate lines.

OUTPUT FORMAT:
Return a SINGLE JSON object containing a list of strings.
{
  "detected_text": [
    "Top header text",
    "Middle meme caption",
    "Another meme text",
    "Bottom punchline text"
  ]
}
"""

    def __init__(self, api_key: str):
        """
        Initialize the Gemini text extractor.

        Args:
            api_key: Google Gemini API key
        """
        genai.configure(api_key=api_key)
        # Use Gemini 1.5 Flash for fast and efficient extraction
        self.model = genai.GenerativeModel("models/gemini-2.5-flash")

    def extract_text(self, image: Image.Image) -> List[str]:
        """
        Extract text from a meme image using Gemini API.

        Args:
            image: PIL Image object

        Returns:
            List of extracted text strings
        """
        try:
            # Generate response from Gemini
            response = self.model.generate_content([self.EXTRACTION_PROMPT, image])

            # Parse the response
            raw_text = response.text
            detected_text = self._parse_response(raw_text)

            return detected_text

        except Exception as e:
            print(f"Error extracting text with Gemini: {e}")
            return []

    def _parse_response(self, raw_text: str) -> List[str]:
        """
        Parse the Gemini API response to extract detected text list.

        Args:
            raw_text: Raw text response from Gemini

        Returns:
            List of detected text strings
        """
        # Clean code blocks if present
        cleaned_text = raw_text.replace("```json", "").replace("```", "").strip()

        try:
            # Try direct JSON parsing first
            parsed = json.loads(cleaned_text)
            if isinstance(parsed, dict) and "detected_text" in parsed:
                return parsed["detected_text"]
        except json.JSONDecodeError:
            pass

        # Fallback: Regex for simple list of strings
        list_match = re.search(r'"detected_text":\s*(\[.*?\])', cleaned_text, re.DOTALL)
        if list_match:
            try:
                # Simple cleanup for common JSON errors
                list_str = list_match.group(1).replace(",]", "]")
                detected_text = json.loads(list_str)
                if isinstance(detected_text, list):
                    return detected_text
            except json.JSONDecodeError:
                pass

        # Last resort: return empty list
        print(f"Could not parse Gemini response: {raw_text[:200]}")
        return []
