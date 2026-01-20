/**
 * WiFi CSI Detection Frontend Application
 * 
 * Handles:
 * - File upload and validation
 * - API communication with FastAPI backend
 * - Visualization of preprocessing and inference results
 */

const API_BASE_URL = 'http://localhost:8000/api';

// ============================================
// DEMO SWITCHING
// ============================================

function switchDemo(demo) {
    document.querySelectorAll('.demo-section').forEach(el => {
        el.classList.remove('active');
    });
    document.getElementById(`${demo}-demo`).classList.add('active');

    document.querySelectorAll('.demo-button').forEach(el => {
        el.classList.remove('active');
    });
    event.target.classList.add('active');
}

// ============================================
// FILE UPLOAD - POSTURE
// ============================================

async function uploadPostureFile(event) {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    showLoading('posture', 'Processing data...');

    try {
        const response = await fetch(`${API_BASE_URL}/posture/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }

        const data = await response.json();
        displayPostureResults(data);
        showSuccess('posture', `Successfully processed ${file.name}`);

    } catch (error) {
        showError('posture', `Error: ${error.message}`);
        console.error('Upload error:', error);
    }
}

// ============================================
// FILE UPLOAD - PRESENCE
// ============================================

async function uploadPresenceFile(event) {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    showLoading('presence', 'Processing data...');

    try {
        const response = await fetch(`${API_BASE_URL}/presence/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }

        const data = await response.json();
        displayPresenceResults(data);
        showSuccess('presence', `Successfully processed ${file.name}`);

    } catch (error) {
        showError('presence', `Error: ${error.message}`);
        console.error('Upload error:', error);
    }
}

// ============================================
// DISPLAY RESULTS - POSTURE
// ============================================

function displayPostureResults(data) {
    if (!data || !data.pipeline) {
        console.error('Invalid response:', data);
        showError('posture', 'Invalid response from server');
        return;
    }
    
    const pipeline = data.pipeline;

    // Step 2: Preprocessing
    if (pipeline.step2_preprocessing) {
        const stats = pipeline.step2_preprocessing.metadata;
        let html = '<div class="stats-grid">';
        html += `<div class="stat-box"><div class="stat-label">Amplitude Mean</div><div class="stat-value">${stats.amplitude_stats.mean.toFixed(2)}</div></div>`;
        html += `<div class="stat-box"><div class="stat-label">Amplitude Std</div><div class="stat-value">${stats.amplitude_stats.std.toFixed(2)}</div></div>`;
        html += `<div class="stat-box"><div class="stat-label">Phase Mean</div><div class="stat-value">${stats.phase_stats.mean.toFixed(2)}</div></div>`;
        html += `<div class="stat-box"><div class="stat-label">Phase Std</div><div class="stat-value">${stats.phase_stats.std.toFixed(2)}</div></div>`;
        html += `<div class="stat-box"><div class="stat-label">Samples</div><div class="stat-value">${stats.batch_size}</div></div>`;
        html += `<div class="stat-box"><div class="stat-label">Subcarriers</div><div class="stat-value">${stats.num_subcarriers}</div></div>`;
        html += '</div>';
        document.getElementById('posture-preprocessing').innerHTML = html;
        document.getElementById('posture-preprocessing').classList.add('active');
    }

    // Step 3: Features
    if (pipeline.step3_feature_extraction) {
        const feat = pipeline.step3_feature_extraction.metadata;
        let html = '<div class="stats-grid">';
        html += `<div class="stat-box"><div class="stat-label">Dimensions</div><div class="stat-value">${feat.feature_shape[1]}</div></div>`;
        html += `<div class="stat-box"><div class="stat-label">Mean</div><div class="stat-value">${feat.feature_stats.mean.toFixed(3)}</div></div>`;
        html += `<div class="stat-box"><div class="stat-label">Std</div><div class="stat-value">${feat.feature_stats.std.toFixed(3)}</div></div>`;
        html += `<div class="stat-box"><div class="stat-label">Min</div><div class="stat-value">${feat.feature_stats.min.toFixed(3)}</div></div>`;
        html += `<div class="stat-box"><div class="stat-label">Max</div><div class="stat-value">${feat.feature_stats.max.toFixed(3)}</div></div>`;
        html += '</div>';
        document.getElementById('posture-features').innerHTML = html;
        document.getElementById('posture-features').classList.add('active');
    }

    // Step 4: Predictions
    if (pipeline.step4_inference && pipeline.step4_inference.inference_result && pipeline.step4_inference.inference_result.predictions) {
        const predictions = pipeline.step4_inference.inference_result.predictions;
        
        if (predictions.length === 0) {
            document.getElementById('posture-predictions').innerHTML = '<div style="color: red;">No predictions available</div>';
            document.getElementById('posture-predictions').classList.add('active');
            return;
        }
        
        // Find most common prediction
        const postureCounts = {};
        let avgConfidence = 0;
        predictions.forEach(pred => {
            postureCounts[pred.posture] = (postureCounts[pred.posture] || 0) + 1;
            avgConfidence += pred.confidence;
        });
        avgConfidence /= predictions.length;
        
        const finalPosture = Object.keys(postureCounts).reduce((a, b) => 
            postureCounts[a] > postureCounts[b] ? a : b
        );
        
        let html = `
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; text-align: center; margin-bottom: 30px;">
                <div style="font-size: 0.9em; opacity: 0.9; margin-bottom: 10px;">FINAL PREDICTION</div>
                <div style="font-size: 2.5em; font-weight: bold; text-transform: uppercase; margin-bottom: 10px;">${finalPosture}</div>
                <div style="font-size: 1.1em;">Confidence: ${(avgConfidence * 100).toFixed(1)}%</div>
            </div>
        `;
        
        html += '<div style="margin-top: 20px;"><div style="font-size: 0.9em; color: #666; margin-bottom: 10px; font-weight: 600;">Individual Sample Predictions:</div>';
        predictions.forEach((pred, idx) => {
            const confidence = (pred.confidence * 100).toFixed(1);
            html += `
                <div class="result-item">
                    <div class="result-label">Sample ${idx + 1}: ${pred.posture}</div>
                    <div class="result-value">Confidence: ${confidence}%</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${confidence}%"></div>
                    </div>
                </div>
            `;
        });
        html += '</div>';
        
        document.getElementById('posture-predictions').innerHTML = html;
        document.getElementById('posture-predictions').classList.add('active');
    }
}

// ============================================
// DISPLAY RESULTS - PRESENCE
// ============================================

function displayPresenceResults(data) {
    if (!data || !data.pipeline) {
        console.error('Invalid response:', data);
        showError('presence', 'Invalid response from server');
        return;
    }
    
    const pipeline = data.pipeline;

    // Step 2: Preprocessing
    if (pipeline.step2_preprocessing) {
        const stats = pipeline.step2_preprocessing.metadata;
        let html = '<div class="stats-grid">';
        html += `<div class="stat-box"><div class="stat-label">RSSI Mean</div><div class="stat-value">${stats.column_stats.rssi.mean.toFixed(1)}</div></div>`;
        html += `<div class="stat-box"><div class="stat-label">Rate Mean</div><div class="stat-value">${stats.column_stats.rate.mean.toFixed(1)}</div></div>`;
        html += `<div class="stat-box"><div class="stat-label">Noise Mean</div><div class="stat-value">${stats.column_stats.noise_floor.mean.toFixed(1)}</div></div>`;
        html += `<div class="stat-box"><div class="stat-label">Channel Mean</div><div class="stat-value">${stats.column_stats.channel.mean.toFixed(1)}</div></div>`;
        html += `<div class="stat-box"><div class="stat-label">Samples</div><div class="stat-value">${stats.batch_size}</div></div>`;
        html += '</div>';
        document.getElementById('presence-preprocessing').innerHTML = html;
        document.getElementById('presence-preprocessing').classList.add('active');
    }

    // Step 3: Features
    if (pipeline.step3_feature_extraction) {
        const feat = pipeline.step3_feature_extraction.metadata;
        let html = '<div class="stats-grid">';
        html += `<div class="stat-box"><div class="stat-label">Dimensions</div><div class="stat-value">${feat.feature_shape[1]}</div></div>`;
        html += `<div class="stat-box"><div class="stat-label">Mean</div><div class="stat-value">${feat.feature_stats.mean.toFixed(3)}</div></div>`;
        html += `<div class="stat-box"><div class="stat-label">Std</div><div class="stat-value">${feat.feature_stats.std.toFixed(3)}</div></div>`;
        html += `<div class="stat-box"><div class="stat-label">Min</div><div class="stat-value">${feat.feature_stats.min.toFixed(3)}</div></div>`;
        html += `<div class="stat-box"><div class="stat-label">Max</div><div class="stat-value">${feat.feature_stats.max.toFixed(3)}</div></div>`;
        html += '</div>';
        document.getElementById('presence-features').innerHTML = html;
        document.getElementById('presence-features').classList.add('active');
    }

    // Step 4: Predictions
    if (pipeline.step4_inference && pipeline.step4_inference.inference_result && pipeline.step4_inference.inference_result.predictions) {
        const predictions = pipeline.step4_inference.inference_result.predictions;
        
        if (predictions.length === 0) {
            document.getElementById('presence-predictions').innerHTML = '<div style="color: red;">No predictions available</div>';
            document.getElementById('presence-predictions').classList.add('active');
            return;
        }
        
        // Count presence/absent
        let presentCount = 0, absentCount = 0;
        let avgConfidence = 0;
        predictions.forEach(pred => {
            if (pred.presence === 'present') presentCount++;
            else absentCount++;
            avgConfidence += pred.confidence;
        });
        avgConfidence /= predictions.length;
        
        const finalPresence = presentCount > absentCount ? 'PRESENT' : 'ABSENT';
        
        let html = `
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; text-align: center; margin-bottom: 30px;">
                <div style="font-size: 0.9em; opacity: 0.9; margin-bottom: 10px;">FINAL DETERMINATION</div>
                <div style="font-size: 2.5em; font-weight: bold; text-transform: uppercase; margin-bottom: 10px;">${finalPresence}</div>
                <div style="font-size: 1.1em;">Detected: ${Math.max(presentCount, absentCount)}/${predictions.length} samples</div>
            </div>
        `;
        
        html += '<div style="margin-top: 20px;"><div style="font-size: 0.9em; color: #666; margin-bottom: 10px; font-weight: 600;">Individual Sample Predictions:</div>';
        predictions.forEach((pred, idx) => {
            const confidence = (pred.confidence * 100).toFixed(1);
            html += `
                <div class="result-item">
                    <div class="result-label">Sample ${idx + 1}: ${pred.presence.toUpperCase()}</div>
                    <div class="result-value">Confidence: ${confidence}%</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${confidence}%"></div>
                    </div>
                </div>
            `;
        });
        html += '</div>';
        
        document.getElementById('presence-predictions').innerHTML = html;
        document.getElementById('presence-predictions').classList.add('active');
    }
}

// ============================================
// UI HELPERS
// ============================================

function showLoading(demo, message) {
    const status = document.getElementById(`${demo}-status`);
    status.innerHTML = `<span class="spinner"></span>${message}`;
    status.className = 'status-message status-loading';
    status.style.display = 'block';
}

function showSuccess(demo, message) {
    const status = document.getElementById(`${demo}-status`);
    status.textContent = message;
    status.className = 'status-message status-success';
    status.style.display = 'block';
}

function showError(demo, message) {
    const status = document.getElementById(`${demo}-status`);
    status.textContent = message;
    status.className = 'status-message status-error';
    status.style.display = 'block';
}

function resetPosture() {
    document.getElementById('posture-file').value = '';
    document.getElementById('posture-status').style.display = 'none';
    document.getElementById('posture-preprocessing').classList.remove('active');
    document.getElementById('posture-features').classList.remove('active');
    document.getElementById('posture-predictions').classList.remove('active');
}

function resetPresence() {
    document.getElementById('presence-file').value = '';
    document.getElementById('presence-status').style.display = 'none';
    document.getElementById('presence-preprocessing').classList.remove('active');
    document.getElementById('presence-features').classList.remove('active');
    document.getElementById('presence-predictions').classList.remove('active');
}

// ============================================
// INITIALIZATION
// ============================================

document.addEventListener('DOMContentLoaded', function() {
    // Setup drag and drop for posture
    const postureUploadArea = document.getElementById('posture-upload-area');
    if (postureUploadArea) {
        postureUploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            postureUploadArea.classList.add('dragover');
        });
        postureUploadArea.addEventListener('dragleave', () => {
            postureUploadArea.classList.remove('dragover');
        });
        postureUploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            postureUploadArea.classList.remove('dragover');
            document.getElementById('posture-file').files = e.dataTransfer.files;
            uploadPostureFile({ target: { files: e.dataTransfer.files } });
        });
    }

    // Setup drag and drop for presence
    const presenceUploadArea = document.getElementById('presence-upload-area');
    if (presenceUploadArea) {
        presenceUploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            presenceUploadArea.classList.add('dragover');
        });
        presenceUploadArea.addEventListener('dragleave', () => {
            presenceUploadArea.classList.remove('dragover');
        });
        presenceUploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            presenceUploadArea.classList.remove('dragover');
            document.getElementById('presence-file').files = e.dataTransfer.files;
            uploadPresenceFile({ target: { files: e.dataTransfer.files } });
        });
    }
});
