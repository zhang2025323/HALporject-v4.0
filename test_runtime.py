import sys
import os
import traceback

os.chdir(r'c:\HALporject\HALporject')
sys.path.insert(0, r'c:\HALporject\HALporject')

print("=" * 60)
print("STEP-BY-STEP IMPORT TEST")
print("=" * 60)

# Step 1: Basic imports
print("\n[1] Testing basic imports...")
try:
    import streamlit as st
    print("    - streamlit OK")
except Exception as e:
    print(f"    - streamlit FAILED: {e}")
    sys.exit(1)

try:
    from PIL import Image
    print("    - PIL OK")
except Exception as e:
    print(f"    - PIL FAILED: {e}")

try:
    import numpy as np
    print("    - numpy OK")
except Exception as e:
    print(f"    - numpy FAILED: {e}")

try:
    from utils.model_loader import Detector
    print("    - model_loader OK")
except Exception as e:
    print(f"    - model_loader FAILED: {e}")
    traceback.print_exc()

# Step 2: Try importing app module (will execute top-level code)
print("\n[2] Testing app.py import (top-level code execution)...")
try:
    # Read and compile the file first
    with open(r'c:\HALporject\HALporject\app.py', 'r', encoding='utf-8') as f:
        code = f.read()
    
    print("    - File read OK")
    print(f"    - File size: {len(code)} bytes")
    
    # Compile to check for syntax errors
    compile(code, 'app.py', 'exec')
    print("    - Compilation OK")
    
    # Execute in controlled environment
    print("    - Executing top-level code...")
    exec(code, {'__name__': '__main__', '__file__': 'app.py'})
    print("    - Execution completed!")
    
except SyntaxError as e:
    print(f"\n[ERROR] SyntaxError at line {e.lineno}:")
    print(f"    {e.text}")
    traceback.print_exc()
    
except ImportError as e:
    print(f"\n[ERROR] Import error: {e}")
    traceback.print_exc()
    
except Exception as e:
    print(f"\n[ERROR] Runtime error during import: {type(e).__name__}")
    print(f"    Message: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST COMPLETED")
print("=" * 60)
