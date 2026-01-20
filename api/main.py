"""
FastAPI Main Application

Provides REST API endpoints for WiFi CSI-based posture and presence detection.
Implements complete pipeline: upload → preprocessing → feature extraction → inference → results

Endpoints:
    POST /api/posture/upload - Upload posture CSV and run inference
    POST /api/presence/upload - Upload presence CSV and run inference
    GET /health - Health check
"""

import logging
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from typing import Dict, Any

from api.inference import InferenceService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="WiFi CSI Detection API",
    description="Research-grade posture and presence detection from WiFi CSI data",
    version="1.0.0"
)

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize inference service
inference_service = InferenceService(model_dir="./models")


# ============================================
# HEALTH CHECK ENDPOINT
# ============================================

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        JSON with service status
    """
    return {
        "status": "healthy",
        "service": "WiFi CSI Detection API",
        "version": "1.0.0",
        "models_available": {
            "posture": inference_service.posture_model is not None,
            "presence": inference_service.presence_model is not None
        }
    }


# ============================================
# POSTURE DETECTION ENDPOINTS
# ============================================

@app.post("/api/posture/upload")
async def upload_posture_data(file: UploadFile = File(...)):
    """
    Upload CSI CSV file and run complete posture detection pipeline.
    
    Expected CSV columns:
    - subcarrier_X_amplitude (X = 1, 2, 3, ...)
    - subcarrier_X_phase (X = 1, 2, 3, ...)
    
    Returns:
        JSON with:
        - metadata: Data info
        - preprocessing: CSI preprocessing visualization data
        - features: Feature extraction data
        - predictions: Posture predictions with confidence
    """
    try:
        # Validate file
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be CSV format")
        
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        logger.info(f"Loaded posture file: {file.filename}, shape: {df.shape}")
        
        # Step 1: Preprocessing
        amplitude_tensor, phase_tensor, features, preprocess_metadata = \
            inference_service.preprocess_posture_data(df)
        
        # Step 2: Feature extraction visualization data
        feature_metadata = {
            "feature_shape": features.shape,
            "feature_stats": {
                "mean": float(np.mean(features)),
                "std": float(np.std(features)),
                "min": float(np.min(features)),
                "max": float(np.max(features))
            },
            "amplitude_tensor_shape": [int(s) for s in amplitude_tensor.shape],
            "phase_tensor_shape": [int(s) for s in phase_tensor.shape]
        }
        
        # Prepare visualization data for preprocessing
        preprocessing_visual = {
            "raw_csi_amplitude": {
                "mean": preprocess_metadata["amplitude_stats"]["mean"],
                "std": preprocess_metadata["amplitude_stats"]["std"],
                "min": preprocess_metadata["amplitude_stats"]["min"],
                "max": preprocess_metadata["amplitude_stats"]["max"],
                "sample_values": amplitude_tensor[0, :min(10, len(amplitude_tensor[0]))].flatten().tolist()
            },
            "processed_csi_amplitude": {
                "sample_values": (amplitude_tensor[0] / (np.std(amplitude_tensor) + 1e-6))[:min(10, len(amplitude_tensor[0]))].flatten().tolist()
            },
            "raw_csi_phase": {
                "mean": preprocess_metadata["phase_stats"]["mean"],
                "std": preprocess_metadata["phase_stats"]["std"],
                "min": preprocess_metadata["phase_stats"]["min"],
                "max": preprocess_metadata["phase_stats"]["max"],
                "sample_values": phase_tensor[0, :min(10, len(phase_tensor[0]))].flatten().tolist()
            },
            "processed_csi_phase": {
                "sample_values": (phase_tensor[0] / (np.std(phase_tensor) + 1e-6))[:min(10, len(phase_tensor[0]))].flatten().tolist()
            }
        }
        
        # Step 3: Model inference
        inference_result = inference_service.infer_posture(features)
        
        # Compile response
        response = {
            "filename": file.filename,
            "status": "success",
            "pipeline": {
                "step1_upload": {
                    "status": "complete",
                    "data_shape": [int(s) for s in df.shape],
                    "num_packets": int(df.shape[0]),
                    "num_subcarriers": len([c for c in df.columns if 'amplitude' in c])
                },
                "step2_preprocessing": {
                    "status": "complete",
                    "metadata": preprocess_metadata,
                    "visualization": preprocessing_visual
                },
                "step3_feature_extraction": {
                    "status": "complete",
                    "metadata": feature_metadata
                },
                "step4_inference": {
                    "status": "complete",
                    "inference_result": inference_result
                }
            }
        }
        
        logger.info(f"Posture detection completed for {file.filename}")
        return JSONResponse(content=response, status_code=200)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing posture file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/posture/preprocess")
async def preprocess_posture_only(file: UploadFile = File(...)):
    """
    Return only preprocessing visualization data without inference.
    Useful for UI step-by-step navigation.
    """
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be CSV format")
        
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        amplitude_tensor, phase_tensor, features, preprocess_metadata = \
            inference_service.preprocess_posture_data(df)
        
        preprocessing_visual = {
            "raw_amplitude_stats": preprocess_metadata["amplitude_stats"],
            "raw_phase_stats": preprocess_metadata["phase_stats"],
            "normalized_amplitude_sample": amplitude_tensor[0].flatten().tolist(),
            "normalized_phase_sample": phase_tensor[0].flatten().tolist()
        }
        
        return JSONResponse(content={
            "status": "success",
            "preprocessing": preprocessing_visual,
            "metadata": preprocess_metadata
        }, status_code=200)
    
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# PRESENCE DETECTION ENDPOINTS
# ============================================

@app.post("/api/presence/upload")
async def upload_presence_data(file: UploadFile = File(...)):
    """
    Upload presence detection CSV file and run inference.
    
    Expected CSV columns:
    - rssi: Signal strength
    - rate: Data rate
    - noise_floor: Noise level
    - channel: WiFi channel
    
    Returns:
        JSON with preprocessing and presence predictions
    """
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be CSV format")
        
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        logger.info(f"Loaded presence file: {file.filename}, shape: {df.shape}")
        
        # Step 1: Preprocessing
        features, preprocess_metadata = inference_service.preprocess_presence_data(df)
        
        # Step 2: Feature visualization
        feature_metadata = {
            "feature_shape": features.shape,
            "feature_stats": {
                "mean": float(np.mean(features)),
                "std": float(np.std(features)),
                "min": float(np.min(features)),
                "max": float(np.max(features))
            }
        }
        
        # Step 3: Model inference
        inference_result = inference_service.infer_presence(features)
        
        # Compile response
        response = {
            "filename": file.filename,
            "status": "success",
            "pipeline": {
                "step1_upload": {
                    "status": "complete",
                    "data_shape": [int(s) for s in df.shape],
                    "num_samples": int(df.shape[0])
                },
                "step2_preprocessing": {
                    "status": "complete",
                    "metadata": preprocess_metadata
                },
                "step3_feature_extraction": {
                    "status": "complete",
                    "metadata": feature_metadata
                },
                "step4_inference": {
                    "status": "complete",
                    "inference_result": inference_result
                }
            }
        }
        
        logger.info(f"Presence detection completed for {file.filename}")
        return JSONResponse(content=response, status_code=200)
    
    except Exception as e:
        logger.error(f"Error processing presence file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# UTILITY ENDPOINTS
# ============================================

@app.get("/api/config")
async def get_config():
    """
    Get system configuration for frontend reference.
    
    Returns:
        JSON with CSI parameters and model info
    """
    from src.config import CSI_CONFIG, POSTURE_MODEL_CONFIG, PRESENCE_MODEL_CONFIG
    
    return {
        "csi_parameters": CSI_CONFIG,
        "posture_model": {
            "num_classes": POSTURE_MODEL_CONFIG['num_classes'],
            "classes": POSTURE_MODEL_CONFIG['classes']
        },
        "presence_model": {
            "type": PRESENCE_MODEL_CONFIG['model_type']
        }
    }
