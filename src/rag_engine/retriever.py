import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class MindGuardRetriever:
    """
    This class acts as the search engine for the RAG pipeline, built on top of
    LangChain's vector store / retriever abstractions. It takes a user's raw
    text, converts it to math, and pulls the most clinically relevant coping
    strategy from our Chroma Vector Database.
    """
    def __init__(self):
        print("🔎 Initializing MindGuard Semantic Retriever (LangChain)...")

        # --- STRICT ARCHITECTURE PATHING ---
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.abspath(os.path.join(self.script_dir, "../../"))
        self.chroma_db_dir = os.path.join(self.project_root, "artifacts", "chroma_db")

        # We MUST use the exact same embedding model used during database creation
        # Otherwise, the search query and the database documents will be on different mathematical maps
        self.embedding_fn = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # --- CONNECT TO THE DATABASE ---
        # We connect to the exact same persistent Chroma collection built by
        # build_vector_db.py, this time wrapped as a LangChain VectorStore.
        # create_collection_if_not_exists=False preserves the original
        # behaviour of failing loudly if the collection was never built.
        self.vector_store = Chroma(
            collection_name="clinical_guidelines",
            embedding_function=self.embedding_fn,
            persist_directory=self.chroma_db_dir,
            create_collection_if_not_exists=False,
        )
        print("✅ Connected to RAG Knowledge Base!")

    def as_retriever(self, emotion_filter=None, n_results=1):
        """
        Builds a native LangChain `VectorStoreRetriever`, ready to drop
        straight into an LCEL chain (e.g. `retriever | format_docs`) anywhere
        else in the project that wants raw LangChain retrieval.
        """
        search_kwargs = {"k": n_results}

        # If our AI Core already diagnosed an emotion, force the retriever to
        # ONLY consider strategies tagged with that emotion.
        if emotion_filter:
            search_kwargs["filter"] = {"emotion": emotion_filter}

        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

    def get_coping_strategy(self, user_query, emotion_filter=None, n_results=1):
        """
        Searches the database for the most relevant strategy.
        Optionally filters by the specific emotion predicted by our XLM-RoBERTa model.
        """
        print(f"\n🧠 Searching Knowledge Base for: '{user_query}'")

        if emotion_filter:
            print(f"🔒 Filtering RAG results strictly for: {emotion_filter}")

        # 1. Build a LangChain retriever scoped to this query's filters
        retriever = self.as_retriever(emotion_filter=emotion_filter, n_results=n_results)

        # 2. Execute the vector search via LangChain's standard retriever interface
        results = retriever.invoke(user_query)

        # 3. Extract and return the raw clinical text
        # LangChain returns a list of `Document` objects; we just want the text.
        if results:
            best_strategy = results[0].page_content
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
