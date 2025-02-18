import chromadb
from sentence_transformers import SentenceTransformer
from config import CHROMA_DB_PATH, EMBEDDER_MODEL

# Initialize ChromaDB and the embedder for semantic retrieval
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
conversation_db = chroma_client.get_or_create_collection(name="chat_memory")
embedder = SentenceTransformer(EMBEDDER_MODEL)

def retrieve_relevant_context(query, top_k=3):
    """Retrieve relevant past messages from ChromaDB using semantic similarity."""
    query_embedding = embedder.encode(query).tolist()
    results = conversation_db.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    retrieved_texts = results.get('documents', [[]])[0]
    return "\n".join(retrieved_texts) if retrieved_texts else ""

def store_conversation(user_input, response):
    """Store the conversation in ChromaDB."""
    conversation_db.add(
        documents=[user_input],
        metadatas=[{"role": "user"}],
        ids=[f"user_{len(conversation_db.get()['documents'])}"]
    )
