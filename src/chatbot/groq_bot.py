import os
import sys
from dotenv import load_dotenv
from groq import Groq

# --- BULLETPROOF IMPORT PATHING ---
# We add the main project folder to Python's system path so we can import our other scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
sys.path.append(project_root)

# Import our custom engines from Phase 1 and Phase 2!
from src.core_model.predict import MindGuardPredictor
from src.rag_engine.retriever import MindGuardRetriever
from src.audio.speech_to_text import MindGuardAudioProcessor
from src.database.db_operations import MindGuardDatabase

class MindGuardChatbot:
    """
    The central intelligence of MindGuard. 
    It orchestrates the emotion prediction, semantic search, and Groq LLM response.
    """
    def __init__(self):
        print("🤖 Booting up MindGuard Conversational Agent...")
        
        # 1. Load API Keys safely
        load_dotenv(os.path.join(project_root, ".env"))
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("❌ GROQ_API_KEY not found in .env file!")
        
        # 2. Initialize the Groq LLM Client
        self.client = Groq(api_key=api_key)
        
        # 3. Wake up our internal tools
        self.predictor = MindGuardPredictor()
        self.retriever = MindGuardRetriever()
        self.audio_processor = MindGuardAudioProcessor()
        self.db = MindGuardDatabase()
        
        # 4. Define the strict rules for the LLM
        self.system_prompt = """
        You are MindGuard, a highly empathetic, clinical-grade mental health AI.
        Your goal is to de-escalate emotional distress and provide actionable coping strategies.
        
        STRICT RULES:
        1. NEVER hallucinate medical advice. ONLY use the 'Clinical Strategy' provided in the prompt.
        2. Keep your response conversational, warm, and easy to read (use short paragraphs).
        3. Do not sound like a robot reading a textbook. Weave the clinical strategy naturally into your empathy.
        4. If the Risk Level is 'High', prioritize grounding the user immediately.
        """
        print("✅ MindGuard Agent is fully operational!")

    def generate_response_from_audio(self, audio_file_path):
        """Pipeline: Listen -> Transcribe -> Predict -> Retrieve -> Generate"""
        print("\n" + "="*50)
        print("🎤 RECEIVING VOICE NOTE...")
        
        # 1. Convert the audio to text using our new module
        transcribed_text = self.audio_processor.transcribe(audio_file_path)
        
        # 2. Pass the transcribed text directly into our existing chatbot pipeline!
        return self.generate_response(user_input=transcribed_text)
    
    def generate_response(self, user_input, session_id="default_user"):
        """The master pipeline: Predict -> Retrieve -> Remember -> Generate -> Save."""
        
        print("\n" + "="*50)
        print(f"👤 USER: {user_input}")
        
        # STEP 1: The Psychologist (Core Model Prediction)
        prediction = self.predictor.predict(user_input)
        emotion = prediction['emotion']
        risk = prediction['risk_level']
        print(f"🧠 DIAGNOSIS: {emotion} (Risk: {risk})")
        
        # STEP 2: The Librarian (RAG Retrieval)
        strategy = self.retriever.get_coping_strategy(
            user_query=user_input, 
            emotion_filter=emotion
        )
        print(f"🗄️ STRATEGY PULLED: {strategy[:50]}...") 
        
        # STEP 3: The Memory Bank (Pull recent history)
        # We grab the last 3 messages so the bot remembers the flow of the conversation
        history = self.db.get_recent_history(session_id=session_id, limit=3)
        history_text = "No previous conversation."
        if history:
            history_text = "\n".join([f"User: {row['user_message']}\nMindGuard: {row['bot_response']}" for row in history])
            
        # STEP 4: Prompt Engineering (Now with Memory!)
        augmented_prompt = f"""
        --- RECENT CONVERSATION HISTORY ---
        {history_text}
        
        --- CURRENT SITUATION ---
        User's New Message: "{user_input}"
        AI Core Diagnosis: {emotion}
        Assessed Risk Level: {risk}
        
        Required Clinical Strategy to Teach the User:
        {strategy}
        
        Draft your response to the user's new message now:
        """
        
        # STEP 5: The Mouth (Groq LLM Generation)
        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": augmented_prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.3, 
        )
        
        final_response = chat_completion.choices[0].message.content
        
        print("\n🤖 MINDGUARD:")
        print(final_response)
        print("="*50)
        
        # STEP 6: Save to Database
        # We silently save this exact interaction so the bot remembers it next time
        self.db.save_interaction(
            session_id=session_id,
            user_message=user_input,
            bot_response=final_response,
            emotion=emotion,
            risk_level=risk
        )
        
        return final_response

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    bot = MindGuardChatbot()
    
    # Let's test the audio pipeline!
    test_audio_path = os.path.join(project_root, "data", "raw", "demo.mpeg")
    
    try:
        bot.generate_response_from_audio(test_audio_path)
    except Exception as e:
        print(f"Error: {e}")