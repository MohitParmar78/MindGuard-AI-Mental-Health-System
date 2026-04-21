import streamlit as st
from components.chat_ui import render_chat
from components.dashboard_ui import render_dashboard

# Set the wide layout for a more professional dashboard look
st.set_page_config(page_title="MindGuard AI", page_icon="🛡️", layout="wide")

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