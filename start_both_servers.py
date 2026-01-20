#!/usr/bin/env python
"""
Start both frontend and backend servers
"""
import subprocess
import sys
import time
import os

os.chdir('e:\\wifi-csi-detection')

# Start backend
print("Starting backend API server on port 8000...")
backend_proc = subprocess.Popen(
    [sys.executable, '-m', 'uvicorn', 'api.main:app', '--host', '0.0.0.0', '--port', '8000'],
    cwd='e:\\wifi-csi-detection'
)

time.sleep(4)

# Start frontend
print("Starting frontend server on port 5000...")
frontend_proc = subprocess.Popen(
    [sys.executable, 'start_frontend.py'],
    cwd='e:\\wifi-csi-detection'
)

print("\n✅ Both servers started!")
print("   Frontend: http://localhost:5000")
print("   Backend:  http://localhost:8000")
print("\nPress Ctrl+C to stop...")

try:
    backend_proc.wait()
    frontend_proc.wait()
except KeyboardInterrupt:
    print("\nStopping servers...")
    backend_proc.terminate()
    frontend_proc.terminate()
    sys.exit(0)
