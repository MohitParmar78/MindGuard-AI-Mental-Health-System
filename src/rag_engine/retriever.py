import os
import chromadb
from chromadb.utils import embedding_functions

class MindGuardRetriever:
    """
    This class acts as the search engine for the RAG pipeline.
    It takes a user's raw text, converts it to math, and pulls the 
    most clinically relevant coping strategy from our Chroma Vector Database.
    """
    def __init__(self):
        print("🔎 Initializing MindGuard Semantic Retriever...")
        
        # --- STRICT ARCHITECTURE PATHING ---
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.abspath(os.path.join(self.script_dir, "../../"))
        self.chroma_db_dir = os.path.join(self.project_root, "artifacts", "chroma_db")
        
        # --- CONNECT TO THE DATABASE ---
        # We connect to the exact same persistent database we built in the previous step
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_db_dir)
        
        # We MUST use the exact same embedding model used during database creation
        # Otherwise, the search query and the database documents will be on different mathematical maps
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        
        # Access the specific collection of clinical guidelines
        self.collection = self.chroma_client.get_collection(
            name="clinical_guidelines",
            embedding_function=self.embedding_fn
        )
        print("✅ Connected to RAG Knowledge Base!")

    def get_coping_strategy(self, user_query, emotion_filter=None, n_results=1):
        """
        Searches the database for the most relevant strategy.
        Optionally filters by the specific emotion predicted by our XLM-RoBERTa model.
        """
        print(f"\n🧠 Searching Knowledge Base for: '{user_query}'")
        
        # 1. Prepare the Search Parameters
        search_kwargs = {
            "query_texts": [user_query],
            "n_results": n_results  # How many strategies to return
        }
        
        # 2. Apply Metadata Filtering (Optional but highly recommended)
        # If our AI Core already diagnosed 'Panic', we can force ChromaDB to ONLY look at Panic strategies
        if emotion_filter:
            print(f"🔒 Filtering RAG results strictly for: {emotion_filter}")
            search_kwargs["where"] = {"emotion": emotion_filter}
            
        # 3. Execute the Vector Search
        results = self.collection.query(**search_kwargs)
        
        # 4. Extract and return the actual text document
        # ChromaDB returns a complex dictionary; we just want the raw clinical text
        if results and results['documents'] and len(results['documents'][0]) > 0:
            best_strategy = results['documents'][0][0]
            print("✅ Found relevant clinical strategy!")
            return best_strategy
        else:
            # Fallback in case the database is empty or the filter is too strict
            print("⚠️ No specific strategy found in database.")
            return "I am here to listen. Could you tell me a little more about how you are feeling?"

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # Instantiate the search engine
    retriever = MindGuardRetriever()
    
    # Simulate a user having a panic attack
    test_query = "I can't breathe, my chest is so tight and the room is spinning."
    
    # We pretend our XLM-RoBERTa model just predicted 'Panic'
    diagnosed_emotion = "Panic"
    
    # Run the retrieval!
    strategy = retriever.get_coping_strategy(
        user_query=test_query, 
        emotion_filter=diagnosed_emotion
    )
    
    print("\n--- RAG Retrieval Result ---")
    print(strategy)