#!/usr/bin/env python
"""
Wrapper to start and keep the API server running
"""
import subprocess
import sys
import time
import os

def main():
    print("=" * 70)
    print("WiFi CSI Detection - API Server Launcher")
    print("=" * 70)
    print("\nStarting backend server on http://0.0.0.0:8000...")
    print("Press Ctrl+C to stop\n")
    
    while True:
        try:
            # Run uvicorn server
            result = subprocess.run(
                [sys.executable, "-m", "uvicorn", "api.main:app", 
                 "--host", "0.0.0.0", "--port", "8000"],
                cwd=os.path.dirname(os.path.abspath(__file__)) or '.'
            )
            
            if result.returncode != 0:
                print("\n⚠️  Server exited with error code:", result.returncode)
                print("Restarting in 5 seconds...")
                time.sleep(5)
            else:
                break
                
        except KeyboardInterrupt:
            print("\n\n✓ Server stopped")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Error starting server: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    main()
