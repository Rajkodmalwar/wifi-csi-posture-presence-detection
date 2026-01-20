#!/usr/bin/env python3
import requests
import json

print("Testing Posture API endpoint...")
try:
    with open('data/sample_posture.csv', 'rb') as f:
        files = {'file': f}
        r = requests.post('http://localhost:8000/api/posture/upload', files=files, timeout=5)
        print(f'Status: {r.status_code}')
        if r.status_code == 200:
            data = r.json()
            # Check structure
            if 'pipeline' in data:
                print("✓ Has pipeline")
                if 'step4_inference' in data['pipeline']:
                    print("✓ Has step4_inference")
                    inf = data['pipeline']['step4_inference']
                    print(f"  Keys: {inf.keys()}")
                    if 'inference_result' in inf:
                        print("✓ Has inference_result")
                        ir = inf['inference_result']
                        print(f"  Keys: {ir.keys()}")
                        if 'predictions' in ir:
                            print(f"✓ Has predictions: {len(ir['predictions'])} samples")
            else:
                print("✗ No pipeline in response")
                print(f"Response keys: {data.keys()}")
        else:
            print(f"✗ Error: {r.text[:200]}")
except Exception as e:
    print(f"✗ Connection error: {e}")

print("\n" + "="*50)
print("Testing Presence API endpoint...")
try:
    with open('data/sample_presence.csv', 'rb') as f:
        files = {'file': f}
        r = requests.post('http://localhost:8000/api/presence/upload', files=files, timeout=5)
        print(f'Status: {r.status_code}')
        if r.status_code == 200:
            data = r.json()
            # Check structure
            if 'pipeline' in data:
                print("✓ Has pipeline")
                if 'step4_inference' in data['pipeline']:
                    print("✓ Has step4_inference")
                    inf = data['pipeline']['step4_inference']
                    print(f"  Keys: {inf.keys()}")
                    if 'inference_result' in inf:
                        print("✓ Has inference_result")
                        ir = inf['inference_result']
                        print(f"  Keys: {ir.keys()}")
                        if 'predictions' in ir:
                            print(f"✓ Has predictions: {len(ir['predictions'])} samples")
            else:
                print("✗ No pipeline in response")
                print(f"Response keys: {data.keys()}")
        else:
            print(f"✗ Error: {r.text[:200]}")
except Exception as e:
    print(f"✗ Connection error: {e}")
