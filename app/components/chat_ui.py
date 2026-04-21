# ============================================================
# FILE: app/components/chat_ui.py
# PURPOSE: Renders the entire Chat Companion page.
#          After every bot response it now:
#            1. Runs SHAP to explain which words drove the emotion
#            2. Reads the predicted emotion + risk level from the DB
#            3. Displays them as colored badges under the message
#            4. Embeds the SHAP HTML report in a collapsible expander
# ============================================================

import streamlit as st                          # Core UI framework
import os                                       # For building file paths
import streamlit.components.v1 as components    # Lets us embed raw HTML (the SHAP report)

# Import our two cached AI loaders from api.py
# get_mindguard_bot()  → returns the chatbot (LLM + Whisper + DB)
# get_shap_explainer() → returns the SHAP XAI engine (XLM-R model)
from api import get_mindguard_bot, get_shap_explainer


# ─────────────────────────────────────────────────────────────
# COLOR CONFIGURATION
# These dictionaries map risk levels and emotion names to
# hex color codes so we can render styled HTML badges.
# ─────────────────────────────────────────────────────────────

# Maps risk level strings → (background_color, text_color) tuples
RISK_COLORS = {
    "High":    ("#ff4b4b", "white"),   # Bright red  — urgent / crisis
    "Medium":  ("#ffa500", "white"),   # Orange      — elevated concern
    "Low":     ("#21c354", "white"),   # Green        — safe / normal
}

# Maps individual emotion label strings → a single hex background color
# Grouped by emotional valence for clarity:
EMOTION_COLORS = {
    # ── Clinical / high-severity emotions (red family) ──────────────
    "Suicidal":             "#ff4b4b",   # Maximum urgency — bright red
    "Depression":           "#e05260",   # Dark rose
    "Anxiety":              "#e07052",   # Warm red-orange
    "Bipolar":              "#c0392b",   # Deep crimson
    "Stress":               "#e67e22",   # Amber-orange
    "Personality disorder": "#9b59b6",   # Purple — complex/clinical

    # ── Positive emotions (green family) ────────────────────────────
    "joy":        "#21c354",
    "love":       "#2ecc71",
    "gratitude":  "#27ae60",
    "admiration": "#1abc9c",
    "optimism":   "#16a085",
    "relief":     "#52be80",
    "excitement": "#58d68d",
    "pride":      "#a9cce3",   # Soft blue-green

    # ── Neutral emotions (blue) ──────────────────────────────────────
    "Normal":  "#3498db",
    "neutral": "#3498db",

    # ── Negative / distressed emotions (amber-brown family) ─────────
    "sadness":        "#e08000",
    "grief":          "#ca6f1e",
    "fear":           "#e74c3c",
    "anger":          "#c0392b",
    "annoyance":      "#d35400",
    "disappointment": "#ca6f1e",
    "remorse":        "#a04000",
    "disgust":        "#7d6608",
}


# ─────────────────────────────────────────────────────────────
# HELPER: Build an HTML emotion badge string
# Returns a colored pill-shaped <span> tag with the emotion name.
# ─────────────────────────────────────────────────────────────
def _emotion_badge(emotion: str) -> str:
    # Look up the color; fall back to grey if emotion isn't in our map
    color = EMOTION_COLORS.get(emotion, "#555555")

    # Build and return a self-contained inline HTML span element
    # unsafe_allow_html=True must be used in the st.markdown call to render this
    return (
        f'<span style="'
        f'background:{color};'          # Background color from our map
        f'color:white;'                 # White text for contrast
        f'padding:3px 10px;'            # Pill-style inner spacing
        f'border-radius:12px;'          # Rounded corners
        f'font-size:13px;'
        f'font-weight:600;'             # Semi-bold text
        f'">🧠 {emotion}</span>'        # Brain emoji + emotion label
    )


# ─────────────────────────────────────────────────────────────
# HELPER: Build an HTML risk-level badge string
# Similar to emotion badge but with icons matching severity.
# ─────────────────────────────────────────────────────────────
def _risk_badge(risk: str) -> str:
    # Unpack background and text color from our tuple map
    bg, fg = RISK_COLORS.get(risk, ("#888888", "white"))

    # Choose a severity icon to reinforce the color signal visually
    icons = {"High": "🚨", "Medium": "⚠️", "Low": "✅"}
    icon = icons.get(risk, "•")   # Default bullet if risk level is unexpected

    return (
        f'<span style="'
        f'background:{bg};'
        f'color:{fg};'
        f'padding:3px 10px;'
        f'border-radius:12px;'
        f'font-size:13px;'
        f'font-weight:600;'
        f'">{icon} Risk: {risk}</span>'
    )


# ─────────────────────────────────────────────────────────────
# HELPER: Render the SHAP HTML report inside the chat message
# Uses streamlit.components.v1.html() to embed arbitrary HTML
# inside a collapsible expander so it doesn't clutter the chat.
# ─────────────────────────────────────────────────────────────
def _render_shap_inline(html_path: str):
    # Only attempt to render if the file actually exists on disk.
    # The file is written by shap_explainer.generate_visual_report().
    if os.path.exists(html_path):
        # Read the entire SHAP HTML file into a string
        with open(html_path, "r", encoding="utf-8") as f:
            shap_html = f.read()

        # Wrap in a Streamlit expander so it's hidden by default.
        # expanded=False means the user must click to open it.
        with st.expander("🔬 View XAI Word-Level Explanation (SHAP)", expanded=False):
            # components.html() injects raw HTML into an iframe inside Streamlit.
            # height=300 sets the iframe height in pixels.
            # scrolling=True enables vertical scroll inside the iframe.
            components.html(shap_html, height=300, scrolling=True)
    else:
        # If no report exists yet, show a subtle placeholder message
        st.caption("_SHAP report not yet generated._")


# ─────────────────────────────────────────────────────────────
# HELPER: Query DB for the most recent emotion + risk level
# Called immediately after bot.generate_response() because the
# bot writes the diagnosed_emotion and risk_level to SQLite during
# response generation. We read it back to display in the UI.
# ─────────────────────────────────────────────────────────────
def _get_last_emotion_risk(bot) -> tuple:
    try:
        # Import the database class from src (path was fixed in api.py's sys.path setup)
        from src.database.db_operations import MindGuardDatabase

        db = MindGuardDatabase()   # Open a new DB connection

        # Run a SQL query to get the single most recent row, ordered newest-first
        db.cursor.execute(
            "SELECT diagnosed_emotion, risk_level "
            "FROM chat_history "
            "ORDER BY timestamp DESC "
            "LIMIT 1"
        )

        row = db.cursor.fetchone()   # Fetch the one result row (or None if empty)
        db.close()                   # Always close the DB connection to avoid leaks

        if row:
            # dict(row) converts the sqlite3.Row object to a regular Python dict
            # so we can access columns by name like a dictionary
            return dict(row)["diagnosed_emotion"], dict(row)["risk_level"]

    except Exception:
        # Silently swallow any DB errors (e.g., table doesn't exist yet)
        # so a DB issue never crashes the entire chat UI
        pass

    # Fallback values if DB is empty or an error occurred
    return "Unknown", "Unknown"


# ─────────────────────────────────────────────────────────────
# MAIN RENDER FUNCTION
# Called by main.py when the user selects "💬 Chat Companion"
# in the sidebar navigation radio buttons.
# ─────────────────────────────────────────────────────────────
def render_chat():
    # Page title and subtitle at the top of the main content area
    st.title("🧠 MindGuard Companion")
    st.markdown("Your clinical-grade, empathetic AI. Type a message or upload a voice note.")

    # ── Load the cached AI objects ────────────────────────────
    # These calls are instant after the first load because of @st.cache_resource
    bot     = get_mindguard_bot()     # The Groq chatbot
    shap_ex = get_shap_explainer()    # The SHAP XAI explainer

    # ── Session State Initialization ──────────────────────────
    # st.session_state persists values across Streamlit reruns
    # within the same browser session. We use it as our "memory".

    if "messages" not in st.session_state:
        # messages: a list of dicts. Each dict has at minimum:
        #   { "role": "user"|"assistant", "content": "..." }
        # Assistant messages may also carry:
        #   { "emotion": "...", "risk": "...", "shap_path": "..." }
        st.session_state.messages = []

    if "session_id" not in st.session_state:
        # session_id is passed to the bot so it can group DB records
        # per user/session. Hardcoded for demo; replace with auth later.
        st.session_state.session_id = "demo_user_001"

    # ── SHAP Report Path ──────────────────────────────────────
    # shap_explainer.py always writes to artifacts/shap_report.html
    # at the project root. We build that path here once so we don't
    # repeat the logic in multiple places.
    SHAP_HTML_PATH = os.path.join(
        # Start from this file's location: app/components/
        # Go up TWO levels (components/ → app/ → project_root/)
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")),
        "artifacts",         # The artifacts/ folder at project root
        "shap_report.html"   # The fixed filename shap_explainer.py writes to
    )

    # ─────────────────────────────────────────────────────────
    # SIDEBAR SECTION
    # st.sidebar renders everything inside it in the left panel.
    # ─────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("🎙️ Voice Input")
        st.write("Feeling overwhelmed? Talk to MindGuard directly.")

        # st.audio_input() renders a mic button in the sidebar.
        # Returns a BytesIO-like object when a recording is complete,
        # or None if no recording has been made yet.
        audio_value = st.audio_input("Record a voice note")

        if audio_value:
            # ── Infinite Loop Prevention ──────────────────────
            # Streamlit reruns on every state change. Without this guard,
            # processing the audio would trigger a rerun, which would
            # process the same audio again — infinitely.
            # FIX: Track the BYTE SIZE of the audio as a unique fingerprint.
            # If the size matches last time, we already processed this recording.
            current_audio_size = len(audio_value.getvalue())

            if ("last_audio_size" not in st.session_state or
                    st.session_state.last_audio_size != current_audio_size):

                # Lock this audio's size into memory to prevent reprocessing
                st.session_state.last_audio_size = current_audio_size

                # Write the audio bytes to a temp .wav file on disk.
                # Whisper needs a file path, not a BytesIO object.
                temp_path = os.path.join("data", "raw", "temp_streamlit.wav")
                with open(temp_path, "wb") as f:
                    f.write(audio_value.getbuffer())   # getbuffer() gives raw bytes

                # Transcribe the audio file using Whisper, then get a response
                with st.spinner("Transcribing and analyzing via Whisper…"):
                    # bot.audio_processor.transcribe() runs Whisper on the .wav file
                    transcribed_text = bot.audio_processor.transcribe(temp_path)
                    # bot.generate_response() sends the text to Groq LLM + writes to DB
                    response = bot.generate_response(
                        user_input=transcribed_text,
                        session_id=st.session_state.session_id
                    )

                # Run SHAP on the transcribed text.
                # This writes a new shap_report.html to artifacts/
                with st.spinner("Generating XAI explanation…"):
                    shap_ex.generate_visual_report(transcribed_text)

                # Pull the emotion + risk level that the bot just saved to the DB
                emotion, risk = _get_last_emotion_risk(bot)

                # Append the user's (transcribed) voice message to the chat history
                st.session_state.messages.append({
                    "role": "user",
                    # Format it visually to indicate it came from voice, not typing
                    "content": f"🎤 **Voice Note:** *{transcribed_text}*"
                })

                # Append the assistant's response WITH the analysis metadata
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "emotion": emotion,          # e.g. "Anxiety"
                    "risk": risk,                # e.g. "High"
                    "shap_path": SHAP_HTML_PATH, # Path to render the SHAP report
                })
                # NOTE: No st.rerun() here. Streamlit will naturally continue
                # down the script and render the new messages in the next section.

        st.divider()   # A horizontal line separator in the sidebar

        # Clear button: wipes all chat history and resets the audio lock
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []                     # Empty the message list
            st.session_state.pop("last_audio_size", None)     # Reset audio fingerprint
            st.rerun()   # Force a full page refresh to visually clear the chat window

    # ─────────────────────────────────────────────────────────
    # MAIN CHAT WINDOW — Render Historical Messages
    # Loop through all messages stored in session_state and
    # re-render them. This is how Streamlit "remembers" the chat.
    # ─────────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        # st.chat_message() renders a chat bubble with the correct
        # avatar: human avatar for "user", robot avatar for "assistant"
        with st.chat_message(msg["role"]):
            # Render the message text (supports markdown formatting)
            st.markdown(msg["content"])

            # Only assistant messages carry the analysis metadata.
            # We check for the "emotion" key to know if this is an
            # analyzed message (not all assistant messages will have it
            # if the DB was empty or an error occurred).
            if msg["role"] == "assistant" and "emotion" in msg:
                # Place the two badges side-by-side using columns
                col1, col2 = st.columns([1, 1])   # Equal-width columns
                with col1:
                    # unsafe_allow_html=True is required because our badge
                    # is a raw HTML string, not standard Markdown
                    st.markdown(_emotion_badge(msg["emotion"]), unsafe_allow_html=True)
                with col2:
                    st.markdown(_risk_badge(msg["risk"]), unsafe_allow_html=True)

                # Render the SHAP explanation report below the badges
                _render_shap_inline(msg.get("shap_path", ""))

    # ─────────────────────────────────────────────────────────
    # MAIN CHAT WINDOW — Handle New Text Input
    # st.chat_input() renders a fixed input bar at the bottom.
    # The walrus operator (:=) assigns AND checks in one step:
    # "if there is a new prompt, assign it to 'prompt' and enter the block"
    # ─────────────────────────────────────────────────────────
    if prompt := st.chat_input("How are you feeling right now?"):

        # 1. Immediately show the user's message (before the bot responds)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Generate bot response + analysis inside the assistant bubble
        with st.chat_message("assistant"):

            # Step A: Get the LLM response (heavy — show spinner to user)
            with st.spinner("Diagnosing emotion and retrieving clinical strategy…"):
                response = bot.generate_response(prompt, st.session_state.session_id)

            # Step B: Run SHAP AFTER the response so the chat feels fast.
            # The user sees the reply first, then waits briefly for XAI.
            with st.spinner("Generating XAI word-level explanation…"):
                # Overwrites artifacts/shap_report.html with a fresh analysis
                shap_ex.generate_visual_report(prompt)

            # Step C: Read the emotion + risk that the bot stored in SQLite
            # during generate_response() — this is a near-instant DB read
            emotion, risk = _get_last_emotion_risk(bot)

            # Step D: Render the response text in the chat bubble
            st.markdown(response)

            # Step E: Render the emotion and risk badges side by side
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(_emotion_badge(emotion), unsafe_allow_html=True)
            with col2:
                st.markdown(_risk_badge(risk), unsafe_allow_html=True)

            # Step F: Embed the SHAP HTML report in a collapsible expander
            _render_shap_inline(SHAP_HTML_PATH)

        # 3. Save the complete message (with metadata) to session_state
        #    so it re-renders correctly on the next Streamlit rerun
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "emotion": emotion,           # Predicted emotion label
            "risk": risk,                 # Risk level (High/Medium/Low)
            "shap_path": SHAP_HTML_PATH,  # Path to the saved SHAP HTML report
        })