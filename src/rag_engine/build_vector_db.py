import os
import json

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class MindGuardVectorDB:
    """
    This class handles the ingestion of clinical guidelines (text)
    and converts them into mathematical embeddings stored in ChromaDB.

    The ingestion pipeline (documents -> embeddings -> vector store) is now
    orchestrated entirely through LangChain: each coping strategy becomes a
    LangChain `Document`, `HuggingFaceEmbeddings` handles the text-to-vector
    conversion, and `Chroma` (from `langchain_chroma`) is the persistent
    vector store — a LangChain-native wrapper around the same ChromaDB
    collection used before.
    """
    def __init__(self):
        print("🗄️ Initializing MindGuard Vector Database Builder (LangChain)...")

        # --- STRICT ARCHITECTURE PATHING ---
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.abspath(os.path.join(self.script_dir, "../../"))

        # Paths aligned perfectly with the folder directory
        self.knowledge_base_path = os.path.join(self.project_root, "data", "knowledge_base", "coping_strategies.json")
        self.chroma_db_dir = os.path.join(self.project_root, "artifacts", "chroma_db")

        # Ensure the Chroma DB output folder exists
        os.makedirs(self.chroma_db_dir, exist_ok=True)

        # --- LANGCHAIN EMBEDDING ENGINE ---
        # Same upgraded model as before (BAAI/bge-base-en-v1.5), now wrapped by
        # LangChain's HuggingFaceEmbeddings so the exact same embedding object
        # can be reused across the ingestion and retrieval layers.
        self.embedding_fn = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # --- LANGCHAIN VECTOR STORE ---
        # `Chroma` from `langchain_chroma` is LangChain's native wrapper around
        # a persistent ChromaDB client. It creates/loads the 'clinical_guidelines'
        # collection for us.
        self.vector_store = Chroma(
            collection_name="clinical_guidelines",
            embedding_function=self.embedding_fn,
            persist_directory=self.chroma_db_dir,
        )
        print(f"✅ Connected to ChromaDB (via LangChain) at: {self.chroma_db_dir}")

    def build_database(self):
        """Reads the JSON file and embeds it into the database as LangChain Documents."""
        print(f"📖 Reading clinical data from: {self.knowledge_base_path}...")

        # 1. Read the JSON file
        with open(self.knowledge_base_path, 'r', encoding='utf-8') as file:
            cbt_data = json.load(file)

        # 2. Wrap every strategy in a LangChain `Document`
        # `page_content` is the text that actually gets embedded and later
        # handed to the LLM; `metadata` travels alongside it for filtering.
        documents = []
        ids = []

        for strategy in cbt_data:
            documents.append(
                Document(
                    page_content=strategy["content"],
                    metadata={
                        "emotion": strategy["primary_emotion"],
                        "risk_level": strategy["target_risk_level"],
                        "category": strategy["category"],
                        "strategy": strategy["strategy_name"],
                        # Chroma metadata values must be str/int/float/bool, so
                        # the tag list becomes a single comma-separated string.
                        "tags": ", ".join(strategy["tags"]),
                    },
                )
            )
            # Reuse the same generalized ID from the JSON instead of a random one
            ids.append(strategy["id"])

        print("⚙️ Embedding text into mathematical vectors via LangChain... (This may take a moment to download the model on the first run)")

        # 3. Inject into the Vector Database through LangChain
        # `add_documents` upserts under the hood, so re-running this script is
        # always safe and never creates duplicate entries.
        self.vector_store.add_documents(documents=documents, ids=ids)

        print(f"✅ Successfully embedded {len(documents)} clinical coping strategies into ChromaDB!")
        print("The RAG Knowledge Base is now primed and ready for the Retriever.")


# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    db_builder = MindGuardVectorDB()
    db_builder.build_database()
