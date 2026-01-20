"""
Mock Model Generator for Demo Testing

Creates dummy pretrained models for demonstration purposes.
These are NOT trained models - they output random predictions.
Use this only for testing the UI/API pipeline.

For real inference, provide actual trained models:
- posture_model.pkl (sklearn model)
- presence_model.pkl (Sklearn trained Random Forest)
- posture_label_encoder.pkl (Sklearn LabelEncoder)
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def create_mock_posture_model(output_dir="./models"):
    """Create a mock posture detection model using Random Forest."""
    
    # Use sklearn Random Forest for posture
    X = np.random.randn(100, 64)  # 64 features
    y = np.random.randint(0, 7, 100)  # 7 classes
    
    model = RandomForestClassifier(n_estimators=20, max_depth=8, random_state=42)
    model.fit(X, y)
    
    output_path = Path(output_dir) / "posture_model.pkl"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Created mock posture model at {output_path}")
    
    return output_path

def create_mock_presence_model(output_dir="./models"):
    """Create a mock presence detection Random Forest model."""
    
    # Create dummy training data
    X = np.random.randn(100, 4)  # 4 features: rssi, rate, noise_floor, channel
    y = np.random.randint(0, 2, 100)  # Binary: present (1) or absent (0)
    
    # Train a simple RF on dummy data
    model = RandomForestClassifier(
        n_estimators=10,
        max_depth=5,
        random_state=42
    )
    model.fit(X, y)
    
    output_path = Path(output_dir) / "presence_model.pkl"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Created mock presence model at {output_path}")
    
    return output_path

def create_mock_label_encoder(output_dir="./models"):
    """Create a mock posture label encoder."""
    
    classes = ['standing', 'sitting', 'lying_down', 'walking', 'running', 'bending', 'arm_raising']
    encoder = LabelEncoder()
    encoder.fit(classes)
    
    output_path = Path(output_dir) / "posture_label_encoder.pkl"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(encoder, f)
    print(f"✓ Created mock label encoder at {output_path}")
    
    return output_path

def main():
    """Generate all mock models."""
    import sys
    
    print("=" * 60)
    print("   WiFi CSI Detection - Mock Model Generator")
    print("=" * 60)
    print("\n⚠️  WARNING: These are DUMMY models for UI testing only!")
    print("   Replace with real trained models for actual inference.\n")
    
    output_dir = "models"
    
    try:
        create_mock_posture_model(output_dir)
        create_mock_presence_model(output_dir)
        create_mock_label_encoder(output_dir)
        
        print("\n" + "=" * 60)
        print("✓ Mock models created successfully!")
        print("=" * 60)
        print(f"\nModels created in: {Path(output_dir).resolve()}")
        print("\nYou can now run the demo:")
        print("  python api/main.py")
        print("  then open: static/index.html in your browser")
        print("\nNote: Predictions from mock models are RANDOM.")
        print("Replace with real trained models for meaningful results.")
        
    except Exception as e:
        print(f"\n❌ Error creating mock models: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
