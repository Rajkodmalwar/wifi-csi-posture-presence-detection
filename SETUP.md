## Installation & Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/wifi-csi-detection.git
cd wifi-csi-detection
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python test_direct_inference.py
```

Expected output:
```
✅ Posture Detection: SUCCESS
✅ Presence Detection: SUCCESS
```

---

## Running the System

### Method 1: Web Interface (Recommended)

```bash
python start_both_servers.py
```

Then open your browser to: **http://localhost:5000**

### Method 2: API Server Only

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

API Documentation: http://localhost:8000/docs

### Method 3: Direct Testing (No Server)

```bash
python test_direct_inference.py
```

---

## Using the Web Interface

1. Select detection type: **Posture** or **Presence**
2. Upload a CSV file:
   - Use `data/sample_posture.csv` for posture
   - Use `data/sample_presence.csv` for presence
3. Click **"Analyze"** to see results
4. View prediction and confidence score

---

## Testing API with curl

### Posture Detection
```bash
curl -X POST \
  -F "file=@data/sample_posture.csv" \
  http://localhost:8000/api/posture/upload
```

### Presence Detection
```bash
curl -X POST \
  -F "file=@data/sample_presence.csv" \
  http://localhost:8000/api/presence/upload
```

### Health Check
```bash
curl http://localhost:8000/health
```

---

## Troubleshooting

**Q: Port 5000 or 8000 already in use**

Change ports in `start_both_servers.py` or use different ports:
```bash
python -m uvicorn api.main:app --port 8001
```

**Q: Models not found**

Ensure models are in `models/` directory:
- `posture_model.pkl`
- `presence_model.pkl`
- `posture_label_encoder.pkl`

**Q: Import errors**

Reinstall dependencies:
```bash
pip install -r requirements.txt --force-reinstall
```

---

## System Structure

```
├── api/                    # FastAPI backend
│   ├── main.py            # REST API endpoints
│   └── inference.py       # Model inference
├── src/                    # Core modules
│   ├── config.py          # Configuration
│   ├── preprocessing/     # Data processing
│   └── model/             # ML models
├── static/                 # Web UI
│   ├── index.html         # Frontend
│   └── app.js             # Logic
├── data/                   # Sample data
├── models/                 # Trained models
├── start_both_servers.py  # Launch everything
├── start_api_server.py    # API only
├── start_frontend.py      # Web UI only
└── README.md              # Main documentation
```

---

## Next Steps

1. ✅ Install and verify
2. ✅ Run the system
3. 📤 Upload sample CSI data
4. 🔍 View predictions
5. 📖 Read the README for more details
6. 🚀 Integrate with your own systems

---

**For more information, see README.md**
