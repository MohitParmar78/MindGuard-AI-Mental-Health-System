# src/database/db_operations.py

import os
import sqlite3

class MindGuardDatabase:
    """
    This class handles the Long-Term Memory of MindGuard.
    It connects to a local SQLite database to save chat logs, track emotions, 
    and pull historical context for the LLM.
    """
    def __init__(self):
        print("💾 Initializing MindGuard SQLite Memory Bank...")
        
        # --- STRICT ARCHITECTURE PATHING ---
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.abspath(os.path.join(self.script_dir, "../../"))
        
        # 1. Define the exact path for the local SQLite file
        # We will save it right next to our ChromaDB in the artifacts folder
        self.db_dir = os.path.join(self.project_root, "artifacts", "database")
        os.makedirs(self.db_dir, exist_ok=True)
        self.db_path = os.path.join(self.db_dir, "mindguard_memory.sqlite3")
        
        try:
            # 2. Establish connection to the local SQLite file
            # check_same_thread=False allows Streamlit/FastAPI to talk to it later without crashing
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            
            # This forces SQLite to return rows as dictionaries, exactly like PostgreSQL did!
            self.conn.row_factory = sqlite3.Row 
            self.cursor = self.conn.cursor()
            print(f"✅ Successfully connected to SQLite at: {self.db_path}")
            
            # 3. Ensure our tables exist
            self._create_tables()
            
        except sqlite3.Error as e:
            print(f"❌ DATABASE ERROR: Could not connect to SQLite.")
            print(f"Details: {e}")

    def _create_tables(self):
        """Creates the database schema if it doesn't already exist."""
        # --- THE FIX: SQLite Syntax ---
        # PostgreSQL uses 'SERIAL' for auto-counting IDs. SQLite uses 'INTEGER PRIMARY KEY AUTOINCREMENT'
        create_table_query = """
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_message TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            diagnosed_emotion TEXT,
            risk_level TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
        self.cursor.execute(create_table_query)
        self.conn.commit() # SQLite requires us to manually commit structural changes
        print("✅ Database schema validated.")

    def save_interaction(self, session_id, user_message, bot_response, emotion, risk_level):
        """Saves a single conversation turn into the database."""
        # --- THE FIX: SQLite Parameters ---
        # PostgreSQL uses %s for security. SQLite uses ?
        insert_query = """
        INSERT INTO chat_history (session_id, user_message, bot_response, diagnosed_emotion, risk_level)
        VALUES (?, ?, ?, ?, ?);
        """
        self.cursor.execute(insert_query, (session_id, user_message, bot_response, emotion, risk_level))
        self.conn.commit() # SQLite requires us to manually commit inserted data
        print("💾 Interaction saved to Long-Term Memory.")

    def get_recent_history(self, session_id, limit=3):
        """
        Pulls the last few messages from a specific user session.
        We will feed this to the Groq LLM so it remembers what it just said.
        """
        select_query = """
        SELECT user_message, bot_response 
        FROM chat_history 
        WHERE session_id = ? 
        ORDER BY timestamp DESC 
        LIMIT ?;
        """
        self.cursor.execute(select_query, (session_id, limit))
        results = self.cursor.fetchall()
        
        # SQLite returns Row objects. We convert them to standard dicts and reverse them.
        return list(reversed([dict(row) for row in results]))

    def close(self):
        """Safely shuts down the database connection."""
        self.cursor.close()
        self.conn.close()
        print("🔌 Database connection closed.")

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    db = MindGuardDatabase()
    
    # Simulate a user session
    test_session = "user_mohit_001"
    
    print("\n--- Testing Database Save ---")
    db.save_interaction(
        session_id=test_session,
        user_message="I have a massive presentation tomorrow and my chest is tight.",
        bot_response="I can sense the tension... start humming a low note.",
        emotion="Nervousness",
        risk_level="Medium"
    )
    
    print("\n--- Testing Database Retrieval ---")
    history = db.get_recent_history(session_id=test_session)
    for row in history:
        print(f"User said: {row['user_message']}")
        print(f"Bot said: {row['bot_response'][:50]}...") # Truncated for readability
        
    db.close()