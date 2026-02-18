# Concatenative Audio Spoofing Detection

A deep learning system for detecting concatenative audio spoofing through frame-level boundary detection. This project identifies manipulated audio segments where real voice samples have been stitched together to create fake audio.

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Evaluation Metrics](#evaluation-metrics)
- [Experimental Results](#experimental-results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Docker Deployment](#docker-deployment)
- [Hardware and Software](#hardware-and-software)
- [Previous Deployment](#previous-deployment)
- [License](#license)
- [Contributing](#contributing)

## Overview

Concatenative audio spoofing involves stitching real voice samples to create realistic-sounding fake audio. This poses significant security threats in voice authentication systems for financial and security applications. Our system detects not only whether audio is spoofed, but also identifies which specific frames are fake and quantifies the degree of manipulation.

### Key Features

- Frame-level detection with 20ms resolution
- Identifies exact boundaries of fake segments
- Quantifies manipulation percentage
- RESTful API for easy integration
- Docker support for scalable deployment
- Weights & Biases integration for experiment tracking

## Problem Statement

### What is Concatenative Audio Spoofing?

Concatenative audio spoofing is a sophisticated attack where segments of genuine voice recordings are cut and stitched together to create fake audio that sounds natural and realistic. Unlike synthetic speech generation, concatenative spoofing uses real voice samples, making it particularly challenging to detect.

### Real-World Threats

- **Voice Authentication Bypass**: Attackers can bypass voice-based security systems in banking and financial applications
- **Identity Fraud**: Creating fake audio evidence for fraudulent activities
- **Social Engineering**: Manipulating voice messages for scams and phishing attacks
- **Misinformation**: Creating fake audio clips of public figures

### Challenges

1. **Natural Sound**: Since real voice samples are used, the audio sounds authentic
2. **Subtle Boundaries**: Concatenation points are often difficult to detect by human ear
3. **Noise Interference**: Background noise can mask manipulation artifacts
4. **Varied Techniques**: Different concatenation methods require robust detection

### Our Solution

This project addresses these challenges through:

1. **Binary Classification** (Solution 1): Determines if entire audio is genuine or spoofed
2. **Frame-Level Detection** (Solution 2 - Current): Identifies exact segments that are fake, providing:
   - Precise localization of manipulated regions
   - Quantification of manipulation (percentage of fake frames)
   - Boundary detection at 20ms resolution

## Model Architecture

The model consists of three main components:

1. **Feature Extractor**: Wav2Vec2-based feature extraction (768-dimensional embeddings)
2. **Frame-Level Embedding**: CNN with 12 residual blocks reducing features to 128 dimensions
3. **Frame-Level Classifier**: Transformer encoder + Bidirectional LSTM for temporal modeling

The architecture processes 6-second audio segments at 16kHz sample rate, producing frame-level predictions with 20ms resolution.

## Dataset

The model is trained on the RFP (Real, Fake, Partially-fake) dataset, a specialized corpus designed for concatenative spoofing detection.

### Dataset Composition

1. **Real Audio Samples**
   - Genuine voice recordings without any manipulation
   - Used as negative examples (label: 0)
   - Provides baseline for authentic speech patterns

2. **Extrinsic Partial Fakes**
   - Audio segments concatenated from different speakers
   - More challenging to create naturally
   - Easier to detect due to voice characteristic changes

3. **Intrinsic Partial Fakes**
   - Audio segments concatenated from the same speaker
   - Sounds more natural and realistic
   - Harder to detect, requires fine-grained analysis

4. **Audio with Noise**
   - Various background noise conditions
   - Tests model robustness in real-world scenarios
   - Includes clean and noisy environments

### Label Format

Labels are provided in text files with temporal annotations:
```
start_time-end_time-authenticity
```
- `start_time`: Beginning of segment (seconds)
- `end_time`: End of segment (seconds)
- `authenticity`: 'R' (Real) or 'F' (Fake)

Example:
```
0.0-2.5-R
2.5-4.0-F
4.0-6.0-R
```

This indicates the audio from 2.5s to 4.0s is fake (concatenated), while the rest is genuine.

## Evaluation Metrics

### Why Equal Error Rate (EER)?

For audio spoofing detection, we use **Equal Error Rate (EER)** as the primary evaluation metric instead of traditional metrics like accuracy or F1-score. Here's why:

#### What is EER?

Equal Error Rate (EER) is the point where the **False Acceptance Rate (FAR)** equals the **False Rejection Rate (FRR)**. In other words, it's where:
- False Positive Rate (FPR) = False Negative Rate (FNR)
- The system makes equal numbers of both types of errors

```
EER = FAR = FRR at threshold θ
```

#### Why Not Accuracy or F1-Score?

1. **Class Imbalance Handling**
   - Audio spoofing datasets often have imbalanced classes
   - Accuracy can be misleading (e.g., 95% accuracy by always predicting "real")
   - F1-score focuses on positive class, but we care equally about both errors

2. **Threshold Independence**
   - EER provides a single metric independent of classification threshold
   - Accuracy and F1 require choosing a specific threshold
   - EER represents the best achievable performance across all thresholds

3. **Security Application Focus**
   - In security systems, both false alarms and missed attacks are critical
   - EER balances both types of errors equally
   - Lower EER means better overall security

4. **Industry Standard**
   - EER is the standard metric in biometric authentication
   - Allows comparison with other spoofing detection systems
   - Used in ASVspoof challenges and research papers

#### EER Calculation

```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(labels, predictions)
fnr = 1 - tpr
eer_threshold = thresholds[np.argmin(np.abs(fnr - fpr))]
eer = fpr[np.argmin(np.abs(fnr - fpr))]
```

#### Interpreting EER

- **Lower is better**: EER of 0% means perfect detection
- **EER of 13%**: System makes 13% errors at optimal threshold
- **Practical meaning**: At EER threshold, 13% of genuine audio is flagged as fake, and 13% of fake audio passes as genuine

### Other Metrics Used

While EER is primary, we also track:

- **Binary Cross-Entropy Loss**: Training objective for frame-level predictions
- **Fake Percentage**: Proportion of frames classified as fake (inference metric)
- **Frame-Level Accuracy**: Per-frame classification accuracy (supplementary)

## Experimental Results

### Solution 1: Binary Classification
Initial approach focused on binary classification of entire audio files.

- **Accuracy**: 85%
- **F1-Score**: 80%
- **Limitation**: Cannot identify which parts of the audio are fake
- **Use Case**: Quick screening, but insufficient for forensic analysis

### Solution 2: Frame-Level Detection (Current Implementation)

Frame-level boundary detection provides precise localization of manipulated segments.

| Feature Extraction Method | EER (Train) | EER (Validation) |
|---------------------------|-------------|------------------|
| Mel Filter Banks          | 18%         | 21%              |
| Wave2Vec (Current)        | **13%**     | **12%**          |

#### Key Findings

1. **Wave2Vec Superiority**
   - 5% improvement over Mel Filter Banks
   - Better generalization (lower validation EER)
   - Captures richer acoustic features

2. **Generalization**
   - Validation EER (12%) slightly better than training (13%)
   - Indicates good model generalization
   - No overfitting observed

3. **Frame-Level Precision**
   - 20ms temporal resolution
   - Identifies exact boundaries of fake segments
   - Enables forensic analysis of manipulation

#### Comparison with Solution 1

| Metric | Solution 1 (Binary) | Solution 2 (Frame-Level) |
|--------|---------------------|--------------------------|
| Detection | Entire audio | Per-frame (20ms) |
| Accuracy | 85% | N/A |
| F1-Score | 80% | N/A |
| EER | Not measured | 12% |
| Localization | No | Yes |
| Forensic Value | Low | High |

## Project Structure

```
.
├── api/                    # FastAPI application
│   ├── __init__.py
│   └── main.py            # API endpoints and server configuration
├── src/                   # Core model implementation
│   ├── __init__.py
│   ├── model.py          # Model architecture definition
│   ├── trainer.py        # Training logic and loops
│   └── inference.py      # Inference pipeline
├── data/                  # Dataset handling
│   ├── __init__.py
│   └── audio_dataset.py  # PyTorch dataset classes
├── utils/                 # Utility functions
│   ├── __init__.py
│   ├── audio_utils.py    # Audio processing utilities
│   ├── metrics.py        # Evaluation metrics (EER calculation)
│   └── logger.py         # Logging configuration
├── config/                # Configuration files
│   ├── __init__.py
│   └── wandb_config.py   # Weights & Biases integration
├── scripts/               # Executable scripts
│   ├── __init__.py
│   ├── run_training.py   # Training script with CLI
│   └── run_inference.py  # Inference script with CLI
├── checkpoints/           # Model checkpoints
├── logs/                  # Training and inference logs
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Docker Compose configuration
├── LICENSE               # MIT License
├── CONTRIBUTING.md       # Contribution guidelines
└── README.md             # This file
```

## Installation

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (optional, for faster inference)
- Docker (optional, for containerized deployment)

### Local Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd AudioSpoofing
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the pre-trained model checkpoint:

The trained model checkpoint must be downloaded separately due to its size.

```bash
# Download from Google Drive
# URL: https://drive.google.com/file/d/1lOg6ZPC9by4ste58zHrzUdhnHqCbf9nD/view?usp=sharing

# Using gdown (recommended)
pip install gdown
gdown 1lOg6ZPC9by4ste58zHrzUdhnHqCbf9nD -O checkpoints/checkpoint_epoch_4.pt

# Or download manually and place in checkpoints/ directory
```

The checkpoint file `checkpoint_epoch_4.pt` must be placed in the `checkpoints/` directory before running inference or starting the API server.

### Environment Variables

Optional environment variables for enhanced functionality:

```bash
export WANDB_API_KEY=your_wandb_api_key  # For experiment tracking
export CHECKPOINT_PATH=checkpoints/checkpoint_epoch_4.pt  # Custom checkpoint path
```

## Usage

### Training

Using the training script:

```bash
python scripts/run_training.py \
    --extrinsic-dir new_files/Extrinsic_Partial_Fakes \
    --intrinsic-dir new_files/Intrinsic_Partial_Fakes \
    --real-dir audio_files/Real/training \
    --label-dir new_files/Generated_Labels \
    --epochs 10 \
    --batch-size 16 \
    --lr 0.001
```

Or directly:

```bash
python -m src.trainer
```

Training logs will be saved to `logs/training_<timestamp>.log` and checkpoints to `checkpoints/`.

### Inference (Standalone)

Using the inference script:

```bash
python scripts/run_inference.py \
    --audio-path path/to/audio.wav \
    --checkpoint-path checkpoints/checkpoint_epoch_4.pt \
    --threshold 0.5
```

Or directly:

```bash
python -m src.inference
```

### API Server

Start the FastAPI server locally:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 7860 --reload
```

Access the interactive API documentation at: http://localhost:7860/docs

## API Documentation

### Endpoints

#### Health Check
```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### Single Audio Prediction
```http
POST /predict
Content-Type: multipart/form-data
```

Parameters:
- `file`: Audio file (WAV, MP3, FLAC, OGG)

Response:
```json
{
  "filename": "sample.wav",
  "fake_percentage": 45.2,
  "total_frames": 299,
  "fake_frames": 135,
  "is_spoofed": false,
  "frame_predictions": [0, 0, 1, 1, 0, ...]
}
```

#### Batch Prediction
```http
POST /predict/batch
Content-Type: multipart/form-data
```

Parameters:
- `files`: Multiple audio files

Response:
```json
{
  "results": [
    {
      "filename": "sample1.wav",
      "fake_percentage": 45.2,
      "is_spoofed": false
    },
    {
      "filename": "sample2.wav",
      "fake_percentage": 78.5,
      "is_spoofed": true
    }
  ]
}
```

### Example Usage with cURL

```bash
# Single prediction
curl -X POST "http://localhost:7860/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio_sample.wav"

# Batch prediction
curl -X POST "http://localhost:7860/predict/batch" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@audio1.wav" \
  -F "files=@audio2.wav"
```

### Example Usage with Python

```python
import requests

url = "http://localhost:7860/predict"
files = {"file": open("audio_sample.wav", "rb")}
response = requests.post(url, files=files)
result = response.json()

print(f"Fake Percentage: {result['fake_percentage']:.2f}%")
print(f"Is Spoofed: {result['is_spoofed']}")
```

## Docker Deployment

### Using Docker

Build and run the container:

```bash
docker build -t audio-spoofing-api .
docker run -p 7860:7860 -v $(pwd)/checkpoints:/app/checkpoints:ro audio-spoofing-api
```

### Using Docker Compose

```bash
docker-compose up -d
```

For GPU support:

```bash
docker-compose up -d
```

View logs:

```bash
docker-compose logs -f
```

Stop the service:

```bash
docker-compose down
```

## Hardware and Software

### Development Environment
- Platform: Lightning AI Online Workspace
- IDE: VS Code
- GPU: Single NVIDIA L4
- Framework: PyTorch 2.0+
- Python: 3.9+

### Deployment
- Previous: Flask application on Hugging Face Spaces
- Current: FastAPI application with Docker support
- Scalable: Supports horizontal scaling with load balancers

## Previous Deployment

This project was previously deployed as a Flask application on Hugging Face Spaces:

https://huggingface.co/spaces/ujalaarshad17/AudioSpoofing

The current version has been migrated to FastAPI for:
- Improved performance and async support
- Automatic interactive API documentation
- Better type validation with Pydantic
- Enhanced scalability for production deployments

## License

This project is open source and available under the MIT License. See the [LICENSE](LICENSE) file for details.

### Open Source

This project is freely available for:
- Academic research
- Commercial applications
- Modification and distribution
- Private use

## Contributing

We encourage contributions from the community. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution

- Model architecture improvements
- Dataset expansion and augmentation
- Performance optimization
- Documentation enhancements
- Bug fixes and testing
- Support for additional audio formats
- Multi-language support

### How to Contribute

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Acknowledgments

- Wav2Vec2 model from Facebook AI Research
- RFP dataset contributors
- Lightning AI for compute resources
- Hugging Face for initial deployment platform
- Open source community for valuable feedback
