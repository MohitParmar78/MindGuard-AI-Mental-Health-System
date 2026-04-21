# ============================================================
# FILE: app/api.py
# PURPOSE: Acts as the central "loader" for all heavy AI objects.
#          Streamlit reruns the entire script on every user action,
#          so we use @st.cache_resource to load models ONCE and
#          reuse the same object for the entire app session.
# ============================================================

import os       # Standard library: used to build file paths safely
import sys      # Standard library: used to modify Python's module search path
import streamlit as st  # Streamlit: the web framework powering the entire UI

# ─────────────────────────────────────────────────────────────
# PATH SETUP BLOCK
# Problem: This file lives inside app/ but our source code lives
#          in src/ (one level up). Python won't find src/ unless
#          we manually tell it where to look.
# ─────────────────────────────────────────────────────────────

# Get the absolute path of THIS file (e.g., /home/user/project/app/api.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go one level UP from app/ to reach the project root (e.g., /home/user/project/)
project_root = os.path.abspath(os.path.join(current_dir, "../"))

# Only add the project root to sys.path if it isn't already there.
# sys.path is the list of directories Python searches when you do "import X".
if project_root not in sys.path:
    sys.path.append(project_root)

# ─────────────────────────────────────────────────────────────
# IMPORTS — now safe because project_root is on sys.path
# ─────────────────────────────────────────────────────────────

# Import our main chatbot class from src/chatbot/groq_bot.py
from src.chatbot.groq_bot import MindGuardChatbot

# Import our SHAP explainability class from src/explainability/shap_explainer.py
from src.explainability.shap_explainer import MindGuardSHAPExplainer


# ─────────────────────────────────────────────────────────────
# CACHED LOADER: MindGuard Chatbot
# ─────────────────────────────────────────────────────────────

@st.cache_resource   # <-- This decorator is the KEY. It tells Streamlit:
                     # "Run this function only ONCE. After that, return the
                     #  same object every time instead of rebuilding it."
                     # Without this, the bot would reload on every keypress.
def get_mindguard_bot():
    """
    Instantiates the MindGuardChatbot and keeps it alive in memory.
    This loads:
      - The Groq LLM connection
      - The Whisper audio transcription model
      - The SQLite database connection
    All of these are expensive to create, so we create them once.
    """
    return MindGuardChatbot()   # Create and return a new bot instance


# ─────────────────────────────────────────────────────────────
# CACHED LOADER: SHAP Explainer
# ─────────────────────────────────────────────────────────────

@st.cache_resource   # Same caching strategy as above — critical here because
                     # MindGuardSHAPExplainer loads a full XLM-RoBERTa model
                     # (~1.1GB weights) which takes ~10-15 seconds to load.
                     # Caching means that cost is paid only once at startup.
def get_shap_explainer():
    """
    Instantiates the MindGuardSHAPExplainer and keeps it alive in memory.
    This loads:
      - XLMRobertaTokenizer (converts text to token IDs)
      - XLMRobertaForSequenceClassification (the 35-emotion neural network)
      - shap.Explainer (the Game Theory math engine wrapped around the model)
    All three are heavy objects — caching is non-negotiable for a smooth UX.
    """
    return MindGuardSHAPExplainer()  # Create and return a new explainer instance