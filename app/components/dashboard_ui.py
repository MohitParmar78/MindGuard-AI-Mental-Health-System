# ============================================================
# FILE: app/components/dashboard_ui.py
# PURPOSE: Renders the Clinical Overview dashboard page.
#          Organized into two tabs:
#            Tab 1 — "📈 Analytics": metrics, charts, history table
#            Tab 2 — "🔬 XAI Report": renders the latest SHAP HTML
#                     report so clinicians can review explanations
#                     without going back to the chat.
# ============================================================

import streamlit as st                          # Core UI framework
import streamlit.components.v1 as components    # For embedding raw HTML (SHAP report)
import pandas as pd                             # For converting SQL rows into a DataFrame
import os                                       # For checking if the SHAP file exists
import sys                                      # For fixing the import path

# ─────────────────────────────────────────────────────────────
# PATH SETUP BLOCK
# Same pattern as api.py: this file is inside app/components/
# so we need to walk up TWO levels to reach the project root
# where the src/ package lives.
# ─────────────────────────────────────────────────────────────

# Get the absolute path of this file (app/components/dashboard_ui.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Walk UP two directories: components/ → app/ → project_root/
project_root = os.path.abspath(os.path.join(current_dir, "../../"))

# Add project_root to Python's search path so "from src.xxx import yyy" works
if project_root not in sys.path:
    sys.path.append(project_root)

# Import our database helper class from src/database/db_operations.py
from src.database.db_operations import MindGuardDatabase


# ─────────────────────────────────────────────────────────────
# SHAP REPORT PATH
# shap_explainer.py ALWAYS writes its output to this exact path.
# We define it once here so any function in this file can use it.
# ─────────────────────────────────────────────────────────────
SHAP_HTML_PATH = os.path.join(
    project_root,       # e.g., /home/user/project/
    "artifacts",        # The artifacts/ folder at project root
    "shap_report.html"  # Fixed filename — shap_explainer.py never changes this
)


# ─────────────────────────────────────────────────────────────
# MAIN RENDER FUNCTION
# Called by main.py when the user selects "📊 Clinical Dashboard"
# in the sidebar navigation.
# ─────────────────────────────────────────────────────────────
def render_dashboard():
    # Page header in the main content area
    st.title("📊 Clinical Overview")
    st.markdown("Real-time emotional tracking, risk assessment, and XAI reports.")

    # ── Create Two Tabs ───────────────────────────────────────
    # st.tabs() returns a list of tab context managers.
    # We unpack them into tab1 and tab2 immediately.
    tab1, tab2 = st.tabs(["📈 Analytics", "🔬 XAI Report"])


    # =========================================================
    # TAB 1: ANALYTICS
    # Shows metrics, bar charts, and a history table built from
    # the chat_history table in our SQLite database.
    # =========================================================
    with tab1:

        # ── Database Read ─────────────────────────────────────
        # Open a database connection to read all historical records
        db = MindGuardDatabase()

        # Execute a SQL SELECT to fetch the three columns we need,
        # sorted newest-first so the table shows recent entries at the top
        db.cursor.execute(
            "SELECT timestamp, diagnosed_emotion, risk_level "
            "FROM chat_history "
            "ORDER BY timestamp DESC"
        )

        # fetchall() returns a list of sqlite3.Row objects (like dicts)
        rows = db.cursor.fetchall()

        # Close the DB connection immediately after reading.
        # Leaving connections open can cause locking issues.
        db.close()

        # Guard clause: if there are no rows in the DB yet,
        # show an info message and exit the function early.
        if not rows:
            st.info("No data yet. Start chatting to generate analytics.")
            return   # Exit render_dashboard() — nothing to display

        # Convert the list of sqlite3.Row objects into a Pandas DataFrame.
        # dict(r) turns each Row into a regular Python dict first.
        # DataFrame() then stacks all those dicts into a table.
        df = pd.DataFrame([dict(r) for r in rows])

        # ── TOP METRICS ROW ───────────────────────────────────
        # Compute the three summary statistics shown as metric cards

        total      = len(df)                              # Total number of chat interactions recorded
        high_risk  = len(df[df["risk_level"] == "High"]) # Count rows where risk_level equals "High"
        # mode()[0] returns the most-frequently-occurring value in that column
        top_emotion = df["diagnosed_emotion"].mode()[0] if not df.empty else "N/A"

        # st.columns(3) creates three equal-width side-by-side columns
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Interactions", total)
        # delta_color="inverse" makes the delta arrow red on increase
        # (since more high-risk flags is BAD, not good)
        col2.metric("High Risk Flags", high_risk, delta_color="inverse")
        col3.metric("Primary Emotion", top_emotion)

        # ── RISK LEVEL SUMMARY ────────────────────────────────
        st.divider()   # Horizontal separator line
        st.subheader("Risk Level Summary")

        # Create three columns, one per risk level
        risk_cols = st.columns(3)

        # Loop through each risk level and display its count
        for i, level in enumerate(["High", "Medium", "Low"]):
            # Count rows where risk_level matches this level
            count = len(df[df["risk_level"] == level])

            # Pick an icon that visually reinforces the severity
            icons = {"High": "🚨", "Medium": "⚠️", "Low": "✅"}

            # Display as a Streamlit metric card in the correct column
            risk_cols[i].metric(f"{icons[level]} {level}", count)

        # ── BAR CHARTS ───────────────────────────────────────
        st.divider()

        # Two charts side by side using columns
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("Emotion Frequency")
            # value_counts() counts how many times each emotion appeared.
            # st.bar_chart() renders it as an interactive bar chart automatically.
            st.bar_chart(df["diagnosed_emotion"].value_counts())

        with chart_col2:
            st.subheader("Risk Level Distribution")
            # Same pattern: count occurrences of each risk level
            st.bar_chart(df["risk_level"].value_counts())

        # ── RECENT HISTORY TABLE ─────────────────────────────
        st.divider()
        st.subheader("Recent Session History")

        # Show only the 20 most recent rows to keep the table manageable.
        # .rename() gives the columns friendly display names for the UI.
        st.dataframe(
            df.head(20).rename(columns={
                "timestamp":         "Time",     # Raw DB column name → human label
                "diagnosed_emotion": "Emotion",
                "risk_level":        "Risk"
            }),
            use_container_width=True   # Stretch table to fill available width
        )


    # =========================================================
    # TAB 2: XAI REPORT
    # Loads and renders the SHAP HTML report that was saved to disk
    # by shap_explainer.generate_visual_report() during the last chat.
    # This gives clinicians a dedicated view of the explanation.
    # =========================================================
    with tab2:
        st.subheader("🔬 Last SHAP Word-Level Explanation")

        # Explanation of what the user is looking at
        st.markdown(
            "This report shows **which words** drove the model's emotion prediction. "
            "**Red highlights** = words that pushed the model TOWARDS that emotion. "
            "**Blue highlights** = words that pushed AGAINST it. "
            "Generated using SHAP (SHapley Additive exPlanations) — Game Theory math."
        )

        # os.path.exists() checks whether the file has been created yet.
        # It won't exist on the very first run before any chat messages.
        if os.path.exists(SHAP_HTML_PATH):

            # Read the entire SHAP HTML report file into a Python string
            with open(SHAP_HTML_PATH, "r", encoding="utf-8") as f:
                shap_html = f.read()

            # components.html() injects the raw HTML string into an iframe
            # inside the Streamlit page. This is how we display SHAP's
            # interactive visualization without needing a Jupyter notebook.
            # height=500 gives enough space to see word highlights clearly.
            # scrolling=True allows vertical scroll if the report is tall.
            components.html(shap_html, height=500, scrolling=True)

            # Show the file path as a small caption below the report
            # so developers can quickly find the file for debugging
            st.caption(f"Report path: `{SHAP_HTML_PATH}`")

        else:
            # No report exists yet — guide the user to generate one
            st.info(
                "No SHAP report found yet. "
                "Send a message in the **💬 Chat Companion** tab "
                "and the XAI report will appear here automatically after your first message."
            )