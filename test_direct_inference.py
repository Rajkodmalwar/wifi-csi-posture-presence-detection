#!/usr/bin/env python
"""
Standalone system test - tests the API by calling inference directly
without needing a running server
"""
import sys
sys.path.insert(0, '.')

from api.inference import InferenceService
import pandas as pd

print("\n" + "="*70)
print("WiFi CSI DETECTION - DIRECT INFERENCE TEST")
print("="*70)

# Initialize service
print("\n[1/3] Initializing Inference Service...")
try:
    inference_service = InferenceService(model_dir="./models")
    print("  ✅ Models loaded successfully")
    print(f"     Posture model: {'✓' if inference_service.posture_model else '✗'}")
    print(f"     Presence model: {'✓' if inference_service.presence_model else '✗'}")
except Exception as e:
    print(f"  ❌ Failed to initialize: {e}")
    sys.exit(1)

# Test posture detection
print("\n[2/3] Testing Posture Detection...")
try:
    # Load sample data
    df_posture = pd.read_csv("data/sample_posture.csv")
    print(f"  Loaded data: {df_posture.shape}")
    
    # Run preprocessing
    amp, phase, features, meta = inference_service.preprocess_posture_data(df_posture)
    print(f"  Preprocessed features shape: {features.shape}")
    
    # Run inference
    result = inference_service.infer_posture(features)
    print(f"  ✅ Posture Detection: SUCCESS")
    if 'predictions' in result and result['predictions']:
        pred = result['predictions'][0]
        print(f"     Prediction: {pred.get('posture')}")
        print(f"     Confidence: {pred.get('confidence', 0)*100:.1f}%")
    else:
        print(f"     Result: {result}")
    
except Exception as e:
    print(f"  ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test presence detection
print("\n[3/3] Testing Presence Detection...")
try:
    # Load sample data
    df_presence = pd.read_csv("data/sample_presence.csv")
    print(f"  Loaded data: {df_presence.shape}")
    
    # Run preprocessing
    features, meta = inference_service.preprocess_presence_data(df_presence)
    print(f"  Preprocessed features shape: {features.shape}")
    
    # Run inference
    result = inference_service.infer_presence(features)
    print(f"  ✅ Presence Detection: SUCCESS")
    if 'predictions' in result and result['predictions']:
        pred = result['predictions'][0]
        print(f"     Prediction: {pred.get('presence')}")
        print(f"     Confidence: {pred.get('confidence', 0)*100:.1f}%")
    else:
        print(f"     Result: {result}")
    
except Exception as e:
    print(f"  ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("✓ Direct inference test complete!")
print("="*70 + "\n")
