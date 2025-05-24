import os
import tempfile
import json
import warnings
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import librosa
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import requests
import sys

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app.logger.setLevel(logging.INFO)

MODEL_NAME = "dima806/bird_sounds_classification"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_cache")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

if not GEMINI_API_KEY:
    app.logger.error("GEMINI_API_KEY environment variable not set.")
    sys.exit(1)

def ensure_model_downloaded():
    if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        app.logger.info(f"Downloading model {MODEL_NAME} to {MODEL_DIR}...")
        try:
            AutoFeatureExtractor.from_pretrained(MODEL_NAME).save_pretrained(MODEL_DIR)
            AutoModelForAudioClassification.from_pretrained(MODEL_NAME).save_pretrained(MODEL_DIR)
            app.logger.info("Model downloaded successfully.")
        except Exception as e:
            app.logger.critical(f"Failed to download model: {e}")
            sys.exit(1)

ensure_model_downloaded()

try:
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_DIR)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_DIR)
    app.logger.info("Model loaded successfully.")
except Exception as e:
    app.logger.critical(f"Failed to load model: {e}")
    sys.exit(1)


def call_gemini_api(species_name: str) -> dict:
    prompt = f"""
    You are a bird expert. Provide Consise JSON-formatted information about the bird species named '{species_name}'. Include these fields exactly:
    - scientificName
    - description
    - habitat
    - diet
    - song
    -available in which countries(in longitude and latitude for heatmap give major exact locations only three places)
    - image (URL if possible)

    Respond ONLY with a JSON object. No extra text, no markdown code blocks.
    """

    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1000
        }
    }

    try:
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        print(data)  # Debugging line to see the raw response

        if 'candidates' in data and data['candidates']:
            content_parts = data['candidates'][0]['content'].get('parts', [])
            if content_parts and 'text' in content_parts[0]:
                raw_content = content_parts[0]['text']
                # Remove markdown if present
                if raw_content.strip().startswith("```json") and raw_content.strip().endswith("```"):
                    raw_content = raw_content.strip()[len("```json"):-3].strip()

                if not raw_content.endswith("}"):
                    app.logger.warning(f"Possibly truncated JSON from Gemini API for {species_name}")
                    raise ValueError("Incomplete JSON received.")

                bird_info = json.loads(raw_content)
                return bird_info
            else:
                raise ValueError("No readable content in Gemini response.")
        else:
            raise ValueError("Invalid Gemini API response structure.")

    except Exception as e:
        app.logger.error(f"Gemini API call error for {species_name}: {e}")
        return {
            "scientificName": species_name,
            "description": "No detailed information available from Gemini API.",
            "habitat": "Unknown",
            "diet": "Unknown",
            "song": "Unknown",
            "image": "https://via.placeholder.com/800x600?text=No+Image"
        }


def get_wikipedia_image(species_name: str) -> str:
    S = requests.Session()
    search_url = "https://en.wikipedia.org/w/api.php"
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": species_name,
        "format": "json",
        "srlimit": 1,
    }
    try:
        response = S.get(search_url, params=search_params)
        data = response.json()
        search_results = data.get("query", {}).get("search", [])
        if not search_results:
            return "https://via.placeholder.com/800x600?text=No+Image"

        page_title = search_results[0]["title"]

        image_query_params = {
            "action": "query",
            "titles": page_title,
            "prop": "pageimages",
            "format": "json",
            "pithumbsize": 800
        }
        response = S.get(search_url, params=image_query_params)
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            thumbnail = page_data.get("thumbnail", {})
            image_url = thumbnail.get("source")
            if image_url:
                return image_url
    except Exception as e:
        app.logger.error(f"Wikipedia image fetch error for {species_name}: {e}")

    return "https://via.placeholder.com/800x600?text=No+Image"


@app.route("/api/classify", methods=["POST"])
def classify_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".wav", ".mp3"]:
        return jsonify({"error": "Invalid file type. Only .wav and .mp3 supported."}), 400

    temp_file = None
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        file.save(temp_file.name)
        temp_file.close()

        waveform, sr = librosa.load(temp_file.name, sr=extractor.sampling_rate)
        inputs = extractor(waveform, sampling_rate=extractor.sampling_rate, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits
            predicted_class_id = torch.argmax(logits).item()
            predicted_label = model.config.id2label[predicted_class_id]

        bird_info = call_gemini_api(predicted_label)
        # Override image URL with Wikipedia image
        bird_info["image"] = get_wikipedia_image(predicted_label)

        response = {
            "prediction": predicted_label,
            "info": bird_info
        }
        return jsonify(response), 200

    except Exception as e:
        app.logger.error(f"Error in classification or Gemini API call: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "model": MODEL_NAME}), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
