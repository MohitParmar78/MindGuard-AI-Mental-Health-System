# src/audio/speech_to_text.py

import os
from dotenv import load_dotenv
from groq import Groq

class MindGuardAudioProcessor:
    """
    This class handles the Multi-Modal Audio component.
    It takes an audio file (mp3, wav, m4a, etc.), sends it to Groq's 
    Whisper-large-v3 model, and returns the transcribed English text.
    """
    def __init__(self):
        print("🎙️ Initializing MindGuard Audio Processing Unit...")
        
        # --- STRICT ARCHITECTURE PATHING ---
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.abspath(os.path.join(self.script_dir, "../../"))
        
        # 1. Load API Keys safely from the .env file
        load_dotenv(os.path.join(self.project_root, ".env"))
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("❌ GROQ_API_KEY not found in .env file!")
        
        # 2. Initialize the Groq Client
        self.client = Groq(api_key=api_key)
        print("✅ Audio Processor is ready to listen!")

    def transcribe(self, audio_file_path):
        """
        Reads a physical audio file and converts the speech to text.
        """
        print(f"\n🎧 Analyzing audio file from: {audio_file_path}...")
        
        # 1. Safety Check: Does the file actually exist?
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"❌ Could not find audio file at: {audio_file_path}")
        
        # 2. Open the file in 'rb' (Read Binary) mode
        with open(audio_file_path, "rb") as file:
            print("⚙️ Transcribing via Groq Whisper-large-v3...")
            
            # 3. Call the Groq Audio API
            transcription = self.client.audio.transcriptions.create(
                file=(audio_file_path, file.read()),
                model="whisper-large-v3",
                prompt="The user is speaking about their mental health, emotions, or stress.", # Gives context to help the AI spell clinical words correctly
                response_format="json",
                language="en", # Forces the output to English
                temperature=0.0 # 0.0 prevents hallucinating words that weren't spoken
            )
            
        # 4. Extract the raw text
        transcribed_text = transcription.text
        
        print("\n📝 --- Transcription Complete ---")
        print(transcribed_text)
        return transcribed_text

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    processor = MindGuardAudioProcessor()
    
    # --- THE FIX: Point exactly to your demo file ---
    test_audio_path = os.path.join(processor.project_root, "data", "raw", "demo.mpeg")
    
    # We use a try/except block so the script doesn't crash if you haven't recorded a file yet
    try:
        text_result = processor.transcribe(test_audio_path)
    except FileNotFoundError as e:
        print("\n⚠️ SETUP REQUIRED FOR TESTING:")
        print("1. Record a short voice memo on your phone or computer (saying you feel stressed).")
        print("2. Save it as 'demo.mpeg' (or .mp3 / .wav).")
        print(f"3. Place it exactly here: {test_audio_path}")
        print("4. Run this script again!")