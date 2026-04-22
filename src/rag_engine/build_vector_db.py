import os
import json
import chromadb
from chromadb.utils import embedding_functions

class MindGuardVectorDB:
    """
    This class handles the ingestion of clinical guidelines (text) 
    and converts them into mathematical embeddings stored in ChromaDB.
    """
    def __init__(self):
        print("🗄️ Initializing MindGuard Vector Database Builder...")
        
        # --- STRICT ARCHITECTURE PATHING ---
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.abspath(os.path.join(self.script_dir, "../../"))
        
        # Paths aligned perfectly with the folder directory
        self.knowledge_base_path = os.path.join(self.project_root, "data", "knowledge_base", "coping_strategies.json")
        self.chroma_db_dir = os.path.join(self.project_root, "artifacts", "chroma_db")
        
        # Ensure the Chroma DB output folder exists
        os.makedirs(self.chroma_db_dir, exist_ok=True)
        
        # --- INITIALIZE CHROMADB ---
        # PersistentClient saves the database directly to your hard drive so you don't lose it when the script stops
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_db_dir)
        
        # Upgraded Embedding Engine (BAAI/bge-base-en-v1.5)
        # This captures significantly more clinical nuance than the default model
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-base-en-v1.5"
        )
        
        # Create or load the 'clinical_guidelines' collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="clinical_guidelines",
            embedding_function=self.embedding_fn
        )
        print(f"✅ Connected to ChromaDB at: {self.chroma_db_dir}")

    def build_database(self):
        """Reads the JSON file and mathematically embeds it into the database."""
        print(f"📖 Reading clinical data from: {self.knowledge_base_path}...")
        
        # 1. Read the JSON file
        with open(self.knowledge_base_path, 'r', encoding='utf-8') as file:
            cbt_data = json.load(file)
            
        # 2. Prepare lists for ChromaDB insertion
        documents = []
        metadatas = []
        ids = []
        
        # 3. Parse the generalized data
        # --- THE FIX: Removed 'enumerate' because we are using our own IDs now ---
        for strategy in cbt_data:
            # The actual text the LLM will read
            documents.append(strategy["content"])
            
            # --- THE FIX: Rich metadata for advanced filtering later ---
            # We map to the exact keys in our new upgraded JSON schema
            metadatas.append({
                "emotion": strategy["primary_emotion"],
                "risk_level": strategy["target_risk_level"],
                "category": strategy["category"],
                "strategy": strategy["strategy_name"],
                # --- THE FIX: Convert the list of tags into a single comma-separated string for ChromaDB ---
                "tags": ", ".join(strategy["tags"]) 
            })
            
            # --- THE FIX: Use our custom generalized ID from the JSON instead of generating a random one ---
            ids.append(strategy["id"])
            
        print("⚙️ Embedding text into mathematical vectors... (This may take a moment to download the model on the first run)")
        
        # 4. Inject into the Vector Database
        # Upsert means "Update or Insert" - it prevents duplicates if you run this script twice
        self.collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Successfully embedded {len(documents)} clinical coping strategies into ChromaDB!")
        print("The RAG Knowledge Base is now primed and ready for the Retriever.")

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    db_builder = MindGuardVectorDB()
    db_builder.build_database()
