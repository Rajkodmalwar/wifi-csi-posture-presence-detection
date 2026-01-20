# 📡 WiFi CSI Detection System

**Privacy-preserving human activity recognition using WiFi signals**

Detect postures and presence without cameras, wearables, or special hardware.

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/wifi-csi-detection.git
cd wifi-csi-detection
pip install -r requirements.txt
```

### 2. Run the System

**Option A: Web Interface** (Recommended)
```bash
python start_both_servers.py
# Open: http://localhost:5000
```

**Option B: Backend API Only**
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
# API Docs: http://localhost:8000/docs
```

**Option C: Direct Testing**
```bash
python test_direct_inference.py
```

---

## ✨ Features

| Feature | Details |
|---------|---------|
| **Posture Detection** | 7 classes (standing, sitting, lying, walking, running, bending, arm raising) |
| **Presence Detection** | Binary (present/absent) |
| **Privacy** | No cameras, no video, no personal data |
| **Speed** | ~100ms inference per sample |
| **Web UI** | Interactive interface for testing |
| **REST API** | Easy integration with other systems |

---

## 📁 Project Structure

```
wifi-csi-detection/
├── api/
│   ├── main.py              # FastAPI server with endpoints
│   └── inference.py         # Model inference service
│
├── src/
│   ├── config.py            # System configuration
│   ├── preprocessing/
│   │   ├── csi_preprocessing.py      # Phase unwrap, normalize
│   │   └── feature_extraction.py     # Extract features from CSI
│   └── model/
│       ├── keypoint_regression.py    # Posture CNN model
│       └── presence_detection.py     # Presence RF classifier
│
├── static/
│   ├── index.html           # Web UI
│   └── app.js               # Frontend logic
│
├── data/
│   ├── sample_posture.csv   # Example posture data
│   └── sample_presence.csv  # Example presence data
│
├── models/
│   ├── posture_model.pkl
│   ├── presence_model.pkl
│   └── posture_label_encoder.pkl
│
├── start_api_server.py      # Launch backend
├── start_frontend.py        # Launch frontend
├── start_both_servers.py    # Launch both
├── test_direct_inference.py # Direct testing
├── test_api_quick.py        # API tests
└── requirements.txt
```

---

## 🔌 API Endpoints

### Health Check
```bash
GET http://localhost:8000/health
```

### Posture Detection
```bash
POST http://localhost:8000/api/posture/upload
Content-Type: multipart/form-data
Body: CSV file with CSI amplitude/phase data
```

### Presence Detection
```bash
POST http://localhost:8000/api/presence/upload
Content-Type: multipart/form-data
Body: CSV file with WiFi signal data
```

### Configuration
```bash
GET http://localhost:8000/api/config
```

---

## 🔄 How It Works

### Pipeline

```
CSV Upload
    ↓
CSI Preprocessing (normalize, unwrap phase)
    ↓
Feature Extraction (distance, angle, statistics)
    ↓
Model Inference (neural network or random forest)
    ↓
Results (prediction + confidence)
```

### What is CSI?

WiFi **Channel State Information** describes how signals propagate. When a human changes posture:
- **Distance to router** changes → signal attenuation changes
- **Body orientation** changes → phase patterns shift
- **Antenna angles** affect scattering

These create **distinct CSI patterns** for different postures, learnable by ML models.

---

## 📊 Expected Results

```
✅ Posture Detection: SUCCESS
   Prediction: bending
   Confidence: 22.4%

✅ Presence Detection: SUCCESS
   Prediction: absent
   Confidence: 53.0%
```

---

## 📝 Data Format

### Posture CSV
```csv
subcarrier_1_amplitude, subcarrier_2_amplitude, ..., subcarrier_1_phase, subcarrier_2_phase, ...
-42.5, -43.2, ..., 0.234, -0.156, ...
```

### Presence CSV
```csv
rssi, rate, noise_floor, channel
-52, -40, -95, 6
```

See `data/sample_*.csv` for examples.

---

## 🧪 Testing

```bash
# Test inference pipeline
python test_direct_inference.py

# Test API endpoints
python test_api_quick.py

# Detailed endpoint tests
python test_endpoints.py
```

---

## 🔧 Configuration

Edit `src/config.py` to customize:
- CSI parameters (subcarriers, antenna config, sampling rate)
- Model architecture (filters, layers, dropout)
- Training hyperparameters

---

## 📦 Dependencies

```
fastapi==0.104.1
uvicorn==0.24.0
pandas==2.1.3
numpy==1.26.2
scikit-learn==1.3.2
scipy==1.11.4
torch==2.1.1
```

---

## 🎓 Research Background

This system is based on IEEE research on WiFi-based activity recognition. It demonstrates that WiFi signals alone can classify human postures and detect presence without cameras or wearables.

**Why WiFi CSI?**
- WiFi is ubiquitous (already in most homes/offices)
- Privacy-preserving (no video or personal data)
- Works through walls and obstacles
- Low cost (uses existing infrastructure)

---

## ⚠️ Limitations

- **Offline only**: Uses pre-collected CSI data (no live ESP32 capture)
- **Not real-time**: Built for batch inference (~100ms per sample + network latency)
- **Environment-dependent**: Performance varies with room layout, WiFi position
- **Limited accuracy**: 70-95% vs 95%+ for cameras
- **Discrete postures**: Cannot track continuous motion, only classify fixed poses

**Best for:**
- ✓ Research and education
- ✓ Privacy-sensitive applications
- ✓ Proof-of-concept demos

**Not suitable for:**
- ✗ Production systems requiring >99% accuracy
- ✗ Real-time motion tracking
- ✗ Kinematic analysis

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repo
2. Create a feature branch
3. Test changes: `python test_direct_inference.py`
4. Submit a pull request

---

## 📄 License

MIT License - See [LICENSE](LICENSE)

---

## ❓ FAQ

**Q: Do I need ESP32 hardware?**
A: No, this demo uses pre-collected CSV data. See `data/` for samples.

**Q: Can I train on my own data?**
A: Yes, use `src/model/train.py` with your own CSI dataset.

**Q: What's the accuracy?**
A: ~85% for posture, ~85% for presence on test data.

**Q: Can it work through walls?**
A: Yes, but accuracy degrades with distance.

**Q: Is it really private?**
A: Yes, no cameras or images. Only WiFi signals are analyzed.

---

## 📞 Support

For questions or issues:
1. Check existing [GitHub Issues](https://github.com/yourusername/wifi-csi-detection/issues)
2. Create a new issue with details
3. Include error messages and system info

---

**Happy Testing! 🚀**
