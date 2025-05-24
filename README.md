# BirdClif Backend API

Backend service for the BirdClif bird species detection system, built with Flask and PyTorch.

## Features

- Bird species detection from audio files
- Integration with Google's Gemini API for detailed species information
- Wikipedia integration for species images
- Geographical distribution data with coordinates
- Real-time audio processing and analysis
- Comprehensive species information including habitat, diet, and behavior

## Dependencies

```plaintext
torch>=2.0.0        # PyTorch for ML model inference
torchaudio>=2.0.0   # Audio processing with PyTorch
flask>=2.0.0        # Web framework
flask-cors>=3.0.0   # Handle CORS
librosa>=0.10.0     # Audio processing
transformers>=4.30.0 # Hugging Face transformers
soundfile>=0.12.0   # Sound file handling
numpy>=1.24.0       # Numerical computations
python-dotenv>=1.0.0 # Environment variable management
gunicorn>=21.2.0    # Production WSGI server
scipy>=1.10.0       # Scientific computations
scikit-learn>=1.3.0 # Machine learning utilities
pandas>=2.0.0       # Data manipulation
```

## Quick Start

```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt

# Start the server
cd "ml model"
python app.py
```

## API Endpoints

### 1. Classify Bird Species

Analyzes an audio file to detect and identify bird species.

**Endpoint:** `/api/classify`

**Method:** POST

**Content-Type:** multipart/form-data

**Request Body:**
```json
{
    "file": "audio_file.mp3"  // Supported formats: MP3, WAV, M4A (max 10MB)
}
```

**Success Response:**
```json
{
    "prediction": "Little Tinamou",
    "info": {
        "scientificName": "Scientific name of the bird",
        "description": "Detailed description of the bird species",
        "habitat": "Natural habitat information",
        "diet": "Dietary preferences",
        "song": "Description of bird's song/call",
        "available in which countries": [
            {
                "country": "India",
                "states": [
                    {
                        "state": "West Bengal",
                        "longitude": "87.854976",
                        "latitude": "22.978650"
                    },
                    {
                        "state": "Kerala",
                        "longitude": "76.2711",
                        "latitude": "10.8505"
                    }
                ]
            }
        ],
        "image": "URL to bird species image"
    }
}
```

**Error Response:**
```json
{
    "error": "Error message description",
    "status": 400
}
```

**Example Usage:**
```python
import requests

url = 'http://localhost:5000/api/classify'
files = {
    'file': open('bird_sound.mp3', 'rb')
}
response = requests.post(url, files=files)
print(response.json())
```

### 2. Health Check

Check if the API server is running properly.

**Endpoint:** `/api/health`

**Method:** GET

**Success Response:**
```json
{
    "status": "healthy",
    "version": "1.0.0",
    "uptime": "2h 30m"
}
```

## API Features

### 1. Audio Analysis
- Supports MP3 and WAV formats
- Machine learning-based species identification
- Real-time processing

### 2. Species Information
- Detailed species descriptions via Gemini API
- Scientific names and classifications
- Habitat and behavioral information
- Dietary preferences
- Song/call descriptions

### 3. Geographical Data
- Location-based distribution data
- Precise coordinates for species sightings
- State/region-wise distribution
- Support for heatmap visualization

### 4. Image Integration
- Automatic species image retrieval from Wikipedia
- High-resolution images when available
- Fallback to placeholder images

## Directory Structure

```
backend/
├── ml model/
│   ├── app.py              # Main Flask application
│   ├── model_cache/        # Pre-trained model files
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── preprocessor_config.json
│   └── uploads/            # Temporary file storage
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Environment Variables

Create a `.env` file in the `ml model` directory:

```env
FLASK_ENV=development
FLASK_APP=app.py
DEBUG=True
MODEL_PATH=model_cache/
UPLOAD_FOLDER=uploads/
MAX_CONTENT_LENGTH=10485760  # 10MB max-size
GEMINI_API_KEY=your_gemini_api_key_here  # Required for species information
```

Make sure to obtain a Google Gemini API key and set it in your environment variables. The API key is required for fetching detailed species information.

## Error Codes

| Code | Description                           |
|------|---------------------------------------|
| 400  | Invalid request (bad file format)     |
| 413  | File too large (>10MB)               |
| 415  | Unsupported media type               |
| 500  | Server error (model inference failed) |

## Model Initialization

The application uses a pretrained bird sound classification model from Hugging Face (`dima806/bird_sounds_classification`). The model files are not included in the repository due to their size. On first startup, the application will automatically download and cache the model (approximately 360MB).

To manually initialize the model:
1. Create a `model_cache` directory: `mkdir "mlmodel/model_cache"`
2. Start the application: `python mlmodel/app.py`
3. The model will be automatically downloaded on first run

Note: Ensure you have enough disk space (at least 500MB) and a stable internet connection for the initial model download.

## Development

1. **Virtual Environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate
   pip install -r requirements.txt
   ```

2. **Running in Debug Mode:**
   ```powershell
   cd "ml model"
   $env:FLASK_ENV = "development"
   $env:FLASK_DEBUG = "1"
   python app.py
   ```

3. **Testing the API:**
   ```powershell
   # Health check
   curl http://localhost:5000/api/health

   # Classify audio (PowerShell)
   $audio = Get-Item "path/to/audio.mp3"
   curl -F "file=@$($audio.FullName)" http://localhost:5000/api/classify
   ```

## Production Deployment

1. **Install Production Server:**
   ```powershell
   pip install gunicorn
   ```

2. **Run with Gunicorn:**
   ```powershell
   cd "ml model"
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

3. **Environment Settings:**
   - Set `FLASK_ENV=production`
   - Disable debug mode
   - Configure proper CORS settings
   - Set up proper logging

## Troubleshooting

1. **Model Loading Issues:**
   - Verify all model files exist in `model_cache/`
   - Check PyTorch/CUDA compatibility
   - Ensure enough system memory

2. **File Upload Issues:**
   - Check file permissions in `uploads/` directory
   - Verify file format and size
   - Check MIME type validation

3. **Performance Issues:**
   - Monitor memory usage
   - Check CPU/GPU utilization
   - Consider batch processing for multiple files

## Security Notes

- Implement rate limiting for API endpoints
- Validate file types before processing
- Clean up uploaded files after processing
- Use secure headers and CORS settings
- Keep dependencies updated
- Monitor for security vulnerabilities
