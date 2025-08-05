#!/usr/bin/env python3
"""
Streamlit Cache Cleaner
This script clears all Streamlit cache and session data.
Run this script if you need to manually clear cache outside of the app.
"""

import os
import shutil
import sys

def clear_streamlit_cache():
    """Clear all Streamlit cache and session data"""
    try:
        print("🧹 Clearing Streamlit cache...")
        
        # Clear Streamlit cache directory
        streamlit_cache_dir = os.path.expanduser("~/.streamlit")
        if os.path.exists(streamlit_cache_dir):
            shutil.rmtree(streamlit_cache_dir, ignore_errors=True)
            print(f"✅ Removed Streamlit cache directory: {streamlit_cache_dir}")
        else:
            print("ℹ️ Streamlit cache directory not found")
        
        # Clear Python cache files in current directory
        current_dir = os.getcwd()
        for root, dirs, files in os.walk(current_dir):
            # Skip virtual environment directories
            if '.venv' in root or '__pycache__' in root:
                continue
                
            for file in files:
                if file.endswith('.pyc'):
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        print(f"✅ Removed: {file_path}")
                    except Exception as e:
                        print(f"⚠️ Could not remove {file_path}: {e}")
        
        # Remove __pycache__ directories in current project
        for root, dirs, files in os.walk(current_dir):
            if '.venv' in root:
                continue
            if '__pycache__' in dirs:
                pycache_path = os.path.join(root, '__pycache__')
                try:
                    shutil.rmtree(pycache_path)
                    print(f"✅ Removed: {pycache_path}")
                except Exception as e:
                    print(f"⚠️ Could not remove {pycache_path}: {e}")
        
        print("🎉 Cache clearing completed!")
        
    except Exception as e:
        print(f"❌ Error clearing cache: {e}")
        sys.exit(1)

if __name__ == "__main__":
    clear_streamlit_cache()