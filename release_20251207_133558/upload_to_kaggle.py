#!/usr/bin/env python3
import os
import json
import subprocess
from pathlib import Path

# Set credentials
os.environ['KAGGLE_USERNAME'] = 'DiyarErol'
os.environ['KAGGLE_API_TOKEN'] = 'KGAT_9b167279a3124d44145a9218c47a1f16'

# Verify credentials file
creds_file = Path.home() / '.kaggle' / 'kaggle.json'
print(f"[1] Checking credentials at {creds_file}")
if creds_file.exists():
    with open(creds_file) as f:
        creds = json.load(f)
    print(f"    ✓ Username: {creds.get('username')}")
    print(f"    ✓ Key: {creds.get('key')[:20]}...")
else:
    print(f"    ✗ Credentials file not found!")

# Check metadata file
metadata_path = Path.cwd() / 'dataset-metadata.json'
print(f"\n[2] Checking metadata at {metadata_path}")
if metadata_path.exists():
    with open(metadata_path) as f:
        metadata = json.load(f)
    print(f"    ✓ Title: {metadata.get('title')}")
    print(f"    ✓ ID: {metadata.get('id')}")
    print(f"    ✓ Subtitle: {metadata.get('subtitle')}")
else:
    print(f"    ✗ Metadata file not found!")

# Try upload
print(f"\n[3] Uploading dataset...")
try:
    result = subprocess.run([
        'kaggle', 'datasets', 'create',
        '-p', '.',
        '-u',
        '--dir-mode=zip'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("    ✓ Upload successful!")
        print(result.stdout)
    else:
        print(f"    ✗ Upload failed!")
        print(f"    Error: {result.stderr}")
        print(f"    Output: {result.stdout}")
except Exception as e:
    print(f"    ✗ Exception: {e}")
