import pandas as pd
import os

def clean_and_merge_data():
    print("Starting data cleaning process...")

    # --- THE FIX: Bulletproof Pathing ---
    # 1. Find exactly where this cleaner.py script lives
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Go up two folders (from src/preprocessing) to find the project root
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))
    
    # 3. Define the exact paths to the data folders
    raw_dir = os.path.join(project_root, "data", "raw")
    processed_dir = os.path.join(project_root, "data", "processed")

    # 1. Process Combined Data.csv
    print("Processing Combined Data.csv...")
    df_combined = pd.read_csv(os.path.join(raw_dir, "Combined Data.csv"))
    df_combined = df_combined[['statement', 'status']].rename(columns={'statement': 'text', 'status': 'label'})

    # 2. Process go_emotions_dataset[1].csv
    print("Processing go_emotions_dataset...")
    df_go = pd.read_csv(os.path.join(raw_dir, "go_emotions_dataset[1].csv"))
    emotion_columns = df_go.columns[3:] 
    df_go['label'] = df_go[emotion_columns].idxmax(axis=1)
    df_go = df_go[['text', 'label']]

    # 3. Process train-00000-of-00001.parquet
    print("Processing parquet file...")
    df_parquet = pd.read_parquet(os.path.join(raw_dir, "train-00000-of-00001.parquet"))
    if 'label' not in df_parquet.columns and 'labels' in df_parquet.columns:
        df_parquet = df_parquet.rename(columns={'labels': 'label'})
    df_parquet = df_parquet[['text', 'label']]

    # 4. Merge all datasets together
    print("Merging datasets...")
    master_df = pd.concat([df_combined, df_go, df_parquet], ignore_index=True)

    # 5. Clean the text formatting
    print("Cleaning text data...")
    master_df['text'] = master_df['text'].astype(str).str.lower().str.strip()

    # 6. Save the final processed dataset
    print("Saving final dataset...")
    output_path = os.path.join(processed_dir, "master_training_data.csv")
    os.makedirs(processed_dir, exist_ok=True)
    master_df.to_csv(output_path, index=False)
    
    print(f"Success! Master dataset created with {len(master_df)} rows and saved to:\n{output_path}")

if __name__ == "__main__":
    clean_and_merge_data()