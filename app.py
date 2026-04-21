# app.py (Root Launcher)
import sys
import os

# Adds the 'app' folder to the search path so imports work correctly
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

# Launches your main Streamlit logic from app/main.py
from app.main import *