#!/usr/bin/env python3
"""
Test script to verify environment variable loading
"""
from dotenv import load_dotenv
import os

print("Testing environment variable loading...")
print("=" * 50)

# Load environment variables
load_dotenv()

# Test the variables
anychat_key = os.getenv("anychat_key")
anychat_model = os.getenv("anychat_model")

print(f"anychat_key: {'[OK] Found' if anychat_key else '[ERROR] Not found'}")
if anychat_key:
    # Show first 8 and last 4 characters for security
    masked_key = anychat_key[:8] + "..." + anychat_key[-4:] if len(anychat_key) > 12 else "***"
    print(f"  Value: {masked_key}")

print(f"anychat_model: {'[OK] Found' if anychat_model else '[ERROR] Not found'}")
if anychat_model:
    print(f"  Value: {anychat_model}")

print("=" * 50)

# Also check if .env file exists and is readable
if os.path.exists('.env'):
    print("[OK] .env file exists")
    with open('.env', 'r') as f:
        content = f.read()
        print(f"[OK] .env file content ({len(content)} characters):")
        print(content)
else:
    print("[ERROR] .env file not found")