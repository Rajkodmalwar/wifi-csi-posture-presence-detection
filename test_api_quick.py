#!/usr/bin/env python
"""
Test the API endpoints
"""
import requests
import sys
import time

print("Connecting to server at http://localhost:8000...")
time.sleep(2)

success_count = 0
total_tests = 3

# Test 1: Health check
print("\n[Test 1] Health Check")
try:
    r = requests.get("http://localhost:8000/health", timeout=5)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        print(f"✅ PASS")
        success_count += 1
    else:
        print(f"❌ FAIL - Status {r.status_code}")
except Exception as e:
    print(f"❌ FAIL - {e}")

# Test 2: Config
print("\n[Test 2] Config Endpoint")
try:
    r = requests.get("http://localhost:8000/api/config", timeout=5)
    if r.status_code == 200:
        data = r.json()
        print(f"Posture classes: {len(data['posture_model']['classes'])}")
        print(f"✅ PASS")
        success_count += 1
    else:
        print(f"❌ FAIL - Status {r.status_code}")
except Exception as e:
    print(f"❌ FAIL - {e}")

# Test 3: File upload
print("\n[Test 3] Posture Detection Upload")
try:
    with open("data/sample_posture.csv", "rb") as f:
        r = requests.post("http://localhost:8000/api/posture/upload",
                         files={"file": ("sample.csv", f)},
                         timeout=10)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        if 'pipeline' in data:
            print(f"Pipeline steps: {list(data['pipeline'].keys())}")
            print(f"✅ PASS")
            success_count += 1
        else:
            print(f"❌ FAIL - No pipeline in response")
    else:
        print(f"❌ FAIL - Status {r.status_code}")
        print(f"Error: {r.text[:200]}")
except Exception as e:
    print(f"❌ FAIL - {e}")

print(f"\n{'='*50}")
print(f"Results: {success_count}/{total_tests} tests passed")
print(f"{'='*50}")

sys.exit(0 if success_count == total_tests else 1)
