import pandas as pd
import os
import torch
import numpy as np                                   # NEW: Needed for math operations on arrays
from torch import nn                                 # NEW: Needed for the custom loss function
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight  # NEW: Calculates the penalty weights
from sklearn.metrics import accuracy_score, f1_score         # NEW: The strict F1 grading system
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

def train_mindguard_model():
    print("🚀 Initializing MindGuard Training Pipeline...")

    # --- BULLETPROOF PATHING ---
    # 1. Find exactly where this train.py script lives
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Go up two folders (from src/core_model) to find the project root
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))
    
    # 3. Define the exact absolute paths
    data_path = os.path.join(project_root, "data", "processed", "master_training_data.csv")
    artifacts_dir = os.path.join(project_root, "artifacts", "xlmr_weights")
    
    # Ensure artifacts directory exists
    os.makedirs(artifacts_dir, exist_ok=True)

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, on_bad_lines='skip')
    
    # --- THE FIX: Pandas parsing safety ---
    df = df.dropna(subset=['text', 'label'])
    df['text'] = df['text'].astype(str)
    df['label'] = df['label'].astype(str)
    
    # --- THE FIX: Data Sanitizer ---
    # 1. Drop any rows where the label is just a number (e.g., "0" or "1")
    df = df[~df['label'].str.isnumeric()]
    # 2. Drop the corrupted 'admi' label
    df = df[df['label'] != 'admi']

    # ⚠️ UNCOMMENT THE LINE BELOW if you don't have a strong GPU and want to do a fast 2-minute test run!
    # df = df.sample(500, random_state=42) 

    # 2. Convert text labels (e.g., 'Anxiety') to numbers (e.g., 0, 1, 2)
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])
    num_labels = len(label_encoder.classes_)
    
    # Save the label mapping
    mapping = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))
    print(f"Detected {num_labels} unique emotions: {mapping}")

    # 3. Split the data into Training (80%) and Testing (20%)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # --- NEW: CALCULATE PENALTY WEIGHTS FOR IMBALANCED DATA ---
    print("⚖️ Calculating Class Weights for Imbalanced Data...")
    unique_classes = np.unique(train_df['label_encoded'])
    weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=train_df['label_encoded'])
    
    # Automatically detect if a GPU is available locally, otherwise use CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    class_weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # 4. Load the XLM-RoBERTa Tokenizer
    print("Loading XLM-RoBERTa Tokenizer...")
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    
    # Function to convert text into numbers
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

    # Apply tokenization to both datasets
    print("Tokenizing the datasets (converting words to numbers)...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)

    # Rename label column so the Trainer understands it
    tokenized_train = tokenized_train.rename_column("label_encoded", "labels")
    tokenized_val = tokenized_val.rename_column("label_encoded", "labels")
    
    # --- THE FIX: Strip out English text so PyTorch only sees numbers ---
    tokenized_train = tokenized_train.remove_columns(["text", "label"])
    tokenized_val = tokenized_val.remove_columns(["text", "label"])
    
    # Formally convert them to PyTorch tensors
    tokenized_train.set_format("torch")
    tokenized_val.set_format("torch")

    # 5. Load the Deep Learning Model
    print("Loading XLM-RoBERTa Neural Network...")
    model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=num_labels)

    # --- NEW: STRICT SCORING METRICS ---
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        # F1 Macro forces the AI to prove it learned the rare emotions, not just 'Normal'
        f1 = f1_score(labels, preds, average='macro') 
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc, 'f1_macro': f1}

    # 6. Set up the Training Rules
    training_args = TrainingArguments(
        output_dir=artifacts_dir,                  # Uses absolute path
        eval_strategy="epoch",                     # Test the model at the end of every round
        learning_rate=3e-5,                        # UPDATED: Slightly higher to help learn rare classes
        per_device_train_batch_size=16,            # How many sentences to look at once
        num_train_epochs=5,                        # UPDATED: 5 epochs to give more time to study hard emotions
        warmup_steps=500,                          # NEW: Gentle warmup to prevent wild guessing early on
        weight_decay=0.01,
        save_strategy="epoch",
        metric_for_best_model="f1_macro",          # NEW: Tell AI to prioritize F1 over basic accuracy
        load_best_model_at_end=True                # NEW: Automatically save the smartest brain
        # overwrite_output_dir=True,
    )

    # --- NEW: CUSTOM TRAINER OVERRIDE ---
    class ImbalancedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # Grab the actual answers
            labels = inputs.pop("labels")
            # Make a prediction
            outputs = model(**inputs)
            logits = outputs.logits
            # Calculate the error using our Custom Penalty Weights!
            loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    # 7. Start Training!
    trainer = ImbalancedTrainer(                   # UPDATED: Using the strict custom trainer
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics            # NEW: Attach the strict grader
    )

    print("🔥 Starting actual model training! (This might take a while depending on your computer)...")
    trainer.train()

    # 8. Save the final model
    final_model_dir = os.path.join(artifacts_dir, "final_mindguard_model")
    print(f"✅ Training complete. Saving the brain to {final_model_dir}...")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

if __name__ == "__main__":
    train_mindguard_model()