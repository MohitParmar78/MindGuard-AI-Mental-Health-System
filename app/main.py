import streamlit as st
import uuid
from components.chat_ui import render_chat
from components.dashboard_ui import render_dashboard

# Set the wide layout for a more professional dashboard look
st.set_page_config(page_title="MindGuard AI", page_icon="🛡️", layout="wide")

# Create one session ID for this browser session. Streamlit reruns this file
# after widget interactions, but session_state survives those reruns. Both the
# chat page and dashboard use this same ID for session isolation.
if "session_id" not in st.session_state:
    st.session_state.session_id = f"streamlit_{uuid.uuid4().hex}"

# Sidebar Navigation
st.sidebar.title("🛡️ MindGuard AI")
st.sidebar.markdown("Welcome to the control panel.")

# Create radio buttons to act as tabs
page = st.sidebar.radio("Navigation", ["💬 Chat Companion", "📊 Clinical Dashboard"])

# Route the user to the correct component based on their selection
if page == "💬 Chat Companion":
    render_chat()
elif page == "📊 Clinical Dashboard":
    render_dashboard()