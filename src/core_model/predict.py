import os
import torch
import torch.nn.functional as F
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

class MindGuardPredictor:
    # The __init__ function runs automatically the moment you create a MindGuardPredictor object.
    def __init__(self, model_path=None):
        print("🧠 Loading MindGuard Core Engine...")
        
        # If no custom path is provided, dynamically find the final model in the artifacts folder.
        if model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(script_dir, "../../"))
            model_path = os.path.join(project_root, "artifacts", "xlmr_weights", "final_mindguard_model")
        
        print(f"Loading weights from: {model_path}")
        
        # Load the vocabulary (Tokenizer) and the trained neural network brain (Model) from your local hard drive.
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
        self.model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
        
        # --- THE FIX: The English Translation Dictionary ---
        # Paste the exact dictionary that printed in your Colab terminal here!
        # This is the 35-emotion dictionary from your earlier local test:
        # --- THE FIX: The English Translation Dictionary ---
        # --- THE FIX: The Final Sanitized Translation Dictionary ---
        # --- The Final Sanitized Translation Dictionary ---
        # Maps the AI's mathematical output (0-34) back to human-readable English words.
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
        
        # Detect if the computer has a GPU ("cuda"), otherwise fall back to standard processor ("cpu").
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Physically move the neural network to the selected hardware.
        self.model.to(self.device)
        # CRITICAL: Lock the model in "evaluation" mode so its weights cannot be accidentally changed during predictions.
        self.model.eval()

    # A clinical triage function to categorize specific emotions into action-oriented risk buckets.
    def determine_risk_level(self, emotion):
        # Standardize the text to lowercase to prevent matching errors (e.g., 'Panic' vs 'panic')
        emotion = emotion.lower()
        high_risk = ['panic', 'severe anxiety', 'depression', 'grief', 'suicidal', 'personality disorder']
        medium_risk = ['stress', 'anxiety', 'anger', 'burnout', 'fear', 'nervousness']
        
        if emotion in high_risk:
            return "High"
        elif emotion in medium_risk:
            return "Medium"
        else:
            return "Low"

    # The core engine function. Takes English text, passes it through the AI, and returns a dictionary of results.
    def predict(self, text):
        # 1. Convert the English sentence into a PyTorch tensor (numbers), padding/truncating it to exactly 128 tokens.
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
        # Move the newly created number tensors to the GPU/CPU to match the model.
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        # 2. Turn off the gradient engine (memory saver) because we are predicting, not training.
        with torch.no_grad(): 
            # Feed the numbers into the neural network.
            outputs = self.model(**inputs)
            # Extract the raw, unformatted mathematical scores for all 35 classes.
            logits = outputs.logits

        # 3. Apply Softmax to convert raw math scores into readable percentages (0.0 to 1.0) that sum to 100%.
        probabilities = F.softmax(logits, dim=-1)
        # Find the single highest percentage (confidence_score) and its corresponding slot number (predicted_class_id).
        confidence_score, predicted_class_id = torch.max(probabilities, dim=-1)

        # --- THE FIX: Translate the math ID back to English ---
        # Extract the pure Python integer from the PyTorch tensor.
        class_id_number = predicted_class_id.item()
        
        # Look up the number in our dictionary. If it can't find it, default to "Unknown"
        predicted_label = self.emotion_map.get(class_id_number, "Unknown")
        
        # Pass the English emotion to our triage function to determine severity.
        risk_level = self.determine_risk_level(predicted_label)

        # Return a cleanly formatted dictionary that a frontend web app or API can easily read.
        return {
            "text": text,
            "emotion": predicted_label,
            "confidence": round(confidence_score.item() * 100, 2),
            "risk_level": risk_level
        }

# --- Quick Test Block ---
# This block only executes if you run this exact file in the terminal. It is ignored if imported elsewhere.
if __name__ == "__main__":
    predictor = MindGuardPredictor()
    sample_text = "I have a massive presentation tomorrow and my chest is tight."
    result = predictor.predict(sample_text)
    
    print("\n--- Prediction Results ---")
    print(f"Input: {result['text']}")
    print(f"Emotion: {result['emotion']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"Risk Level: {result['risk_level']}")