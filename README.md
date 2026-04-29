# 🔍 Deepfake Detection System

A production-grade deepfake detection system using **EfficientNet-B4 + Frequency Analysis + CBAM Attention**.

---

## 📁 Project Structure

```
deepfake-detection/
├── models/
│   └── detector.py          # Core model: EfficientNet + Attention + Freq branch
├── src/
│   ├── train.py             # Training pipeline with mixed precision + focal loss
│   └── inference.py         # Inference engine + Grad-CAM explainability
├── utils/
│   ├── evaluate.py          # Metrics, ROC/PR curves, evaluation
│   └── preprocessing.py     # Frame extraction, dataset prep, augmentation
├── tests/
│   └── test_all.py          # Unit tests for all modules
├── app.py                   # FastAPI REST API
├── main.py                  # CLI entry point
└── requirements.txt
```

---

## 🏗️ Architecture

```
Input Image (224×224×3)
        │
        ├──── EfficientNet-B4 Backbone
        │            │
        │       CBAM Attention          ← Channel + Spatial attention
        │            │
        │      Global Avg Pool → [1792-d]
        │
        └──── Frequency Branch          ← FFT magnitude → CNN
                     │
                [256-d freq features]
        
                     ↓
              [ Concatenate 2048-d ]
                     ↓
              Fully Connected → Sigmoid
                     ↓
              P(fake) ∈ [0, 1]
```

**Key design choices:**
- **EfficientNet-B4** — strong spatial feature extractor pretrained on ImageNet
- **Frequency Branch** — detects GAN-specific artifacts in the DCT/FFT domain
- **CBAM** — localizes forgery regions for better classification & explainability
- **Focal Loss** — handles the real/fake class imbalance
- **Grad-CAM** — visual explanation of what the model focuses on

---

## ⚙️ Setup

```bash
# 1. Clone and install
git clone <repo>
cd deepfake-detection
pip install -r requirements.txt

# 2. Prepare your dataset
#    Expected layout:
#    data/
#      real/   ← real face images (.jpg/.png)
#      fake/   ← deepfake images

python main.py prepare --data_dir data/
```

---

## 🚀 Training

```bash
python main.py train \
  --data_dir data/ \
  --epochs 30 \
  --batch_size 32 \
  --lr 1e-4 \
  --save_dir checkpoints/
```

**Training features:**
- Mixed-precision training (AMP) for 2× speedup
- Cosine LR scheduler with warmup
- Layer-wise LR decay (backbone LR = head LR × 0.1)
- Weighted sampling for class imbalance
- Best model saved by Val AUC

---

## 📊 Evaluation

```bash
python main.py eval \
  --data_dir data/ \
  --model checkpoints/best_model.pth \
  --output_dir results/
```

Outputs: AUC, F1, Accuracy, Precision, Recall, ROC curve, PR curve.

---

## 🔍 Inference

**Single image:**
```bash
python main.py predict \
  --input path/to/face.jpg \
  --model checkpoints/best_model.pth \
  --save_heatmap heatmap.jpg
```

**Video:**
```bash
python main.py predict \
  --input path/to/video.mp4 \
  --model checkpoints/best_model.pth
```

**Python API:**
```python
from src.inference import DeepfakeInference

engine = DeepfakeInference(model_path="checkpoints/best_model.pth")
result = engine.predict_image("face.jpg")

print(result.label)        # "REAL" or "FAKE"
print(result.probability)  # e.g. 0.9341
print(result.confidence)   # certainty 0-1
# result.heatmap → numpy array for Grad-CAM overlay
```

---

## 🌐 REST API

```bash
python main.py serve --port 8000
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/detect/image` | Analyze uploaded image |
| POST | `/api/detect/video` | Analyze uploaded video |
| GET  | `/api/health`       | Health check |
| GET  | `/api/stats`        | Detection statistics |

**Example curl:**
```bash
curl -X POST http://localhost:8000/api/detect/image \
  -F "file=@face.jpg" \
  -F "return_heatmap=true"
```

**Response:**
```json
{
  "label": "FAKE",
  "probability": 0.9341,
  "confidence": 0.8682,
  "is_fake": true,
  "heatmap_b64": "...",
  "processing_time_ms": 142.5
}
```

---

## 🧪 Testing

```bash
pytest tests/ -v
```

---

## 📦 Recommended Datasets

| Dataset | Size | Notes |
|---------|------|-------|
| [FaceForensics++](https://github.com/ondyari/FaceForensics) | ~1M frames | Gold standard, 4 manipulation methods |
| [DFDC (Facebook)](https://ai.facebook.com/datasets/dfdc/) | 128k videos | Large-scale, diverse |
| [Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics) | 590 videos | High-quality deepfakes |
| [DFD (Google)](https://ai.googleblog.com/2019/09/contributing-data-to-deepfake.html) | 3k videos | Consented actors |

---

## 📈 Expected Performance (FaceForensics++ c23)

| Metric | Value |
|--------|-------|
| AUC    | ~0.97 |
| Accuracy | ~95% |
| F1     | ~0.94 |

---

## 🛠️ Extending the System

**Add a new backbone:**
```python
# In models/detector.py, replace:
efficientnet = models.efficientnet_b4(...)
# With any torchvision model
```

**Custom augmentation:**
```python
from utils.preprocessing import DeepfakeAugmentor
aug = DeepfakeAugmentor()
augmented = aug.apply_random(img, p=0.4)
```

**Ensemble inference:**
```python
from models.detector import EnsembleDetector
ensemble = EnsembleDetector(["model1.pth", "model2.pth"])
result = ensemble.predict(batch_tensor)
```
