# src/explainability/shap_explainer.py

import os
import torch
import shap
from transformers import pipeline, XLMRobertaTokenizer, XLMRobertaForSequenceClassification

class MindGuardSHAPExplainer:
    """
    This class handles the Explainable AI (XAI) component of MindGuard.
    It uses Game Theory (SHAP values) to mathematically prove exactly 
    which words caused the neural network to predict a specific emotion.
    """
    def __init__(self):
        print("🔍 Initializing MindGuard SHAP Explainability Engine...")
        
        # --- STRICT ARCHITECTURE PATHING ---
        # 1. Locate the current script (src/explainability/shap_explainer.py)
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Traverse up TWO directories to hit the MINDGUARD_AI_PROJECT root
        self.project_root = os.path.abspath(os.path.join(self.script_dir, "../../"))
        
        # 3. Define the exact path to where we saved the trained brain in Day 3
        self.model_path = os.path.join(self.project_root, "artifacts", "xlmr_weights", "final_mindguard_model")
        
        # 4. Define where the visual HTML reports will be saved
        self.artifacts_dir = os.path.join(self.project_root, "artifacts")
        
        # --- THE FIX 1: The Translation Dictionary ---
        # We must define the 35 English emotions so SHAP doesn't output "LABEL_X"
        self.emotion_map = {
            0: 'Anxiety', 1: 'Bipolar', 2: 'Depression', 3: 'Normal', 
            4: 'Personality disorder', 5: 'Stress', 6: 'Suicidal', 7: 'admiration', 
            8: 'amusement', 9: 'anger', 10: 'annoyance', 11: 'approval', 
            12: 'caring', 13: 'confusion', 14: 'curiosity', 15: 'desire', 
            16: 'disappointment', 17: 'disapproval', 18: 'disgust', 19: 'embarrassment', 
            20: 'excitement', 21: 'fear', 22: 'gratitude', 23: 'grief', 
            24: 'joy', 25: 'love', 26: 'nervousness', 27: 'neutral', 
            28: 'optimism', 29: 'pride', 30: 'realization', 31: 'relief', 
            32: 'remorse', 33: 'sadness', 34: 'surprise'
        }

        print(f"Loading Core Brain from: {self.model_path}...")
        
        # --- LOAD THE AI CORE ---
        # Initialize the tokenizer (translates text to numbers) and the model (the brain)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(self.model_path)
        self.model = XLMRobertaForSequenceClassification.from_pretrained(self.model_path)
        
        # --- THE FIX 2: Inject the Dictionary into the Model's Brain ---
        # This permanently forces the Hugging Face model to speak English instead of Math
        self.model.config.id2label = self.emotion_map
        self.model.config.label2id = {v: k for k, v in self.emotion_map.items()}

        # Use Hugging Face's pipeline to wrap the model for easy SHAP integration
        # Set device to 0 if a GPU is detected, otherwise fallback to CPU (-1)
        self.device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline(
            "text-classification", 
            model=self.model, 
            tokenizer=self.tokenizer, 
            device=self.device, 
            top_k=None # top_k=None forces the AI to output scores for ALL 35 emotions, not just the top guess
        )
        
        # --- WARM UP SHAP ---
        print("⚙️ Warming up Game Theory Math (SHAP)...")
        # Pass our classifier pipeline into the SHAP Explainer engine
        self.explainer = shap.Explainer(self.classifier)
        print("✅ SHAP Explainer ready!")

    def generate_visual_report(self, text):
        """
        Takes a raw string of text, runs it through the model, 
        calculates SHAP values, and outputs an interactive HTML file.
        """
        print(f"\n🧠 Analyzing: '{text}'")
        
        # 1. Run the Game Theory calculations
        # This isolates the impact of every single word on the final prediction
        shap_values = self.explainer([text])
        
        # 2. Define the exact save location for the HTML report
        html_path = os.path.join(self.artifacts_dir, "shap_report.html")
        
        # --- THE FIX 3: Targeted Slicing ---
        # Instead of drawing 35 overlapping arrows, find the emotion the AI was MOST confident in.
        best_class_index = shap_values[0].values.sum(axis=0).argmax()
        
        # 3. Generate the visualization ONLY for the winning emotion
        # display=False ensures it generates the raw HTML instead of trying to open a Jupyter widget
        shap_html = shap.plots.text(shap_values[0, :, best_class_index], display=False)
        
        # 4. Save the HTML string to a physical file in the artifacts folder
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(shap_html)
            
        print(f"✅ Diagnostic Complete!")
        print(f"Visual Report saved to: {html_path}")
        print("Go to your 'artifacts' folder and open 'shap_report.html' in your browser.")

# --- EXECUTION BLOCK ---
# This block only runs if this specific file is executed directly from the terminal
if __name__ == "__main__":
    # Instantiate our explainer class
    explainer = MindGuardSHAPExplainer()
    
    # Define a test patient input
    sample_text = "I have a massive presentation tomorrow and my chest is tight."
    
    # Generate the explanation report
    explainer.generate_visual_report(sample_text)