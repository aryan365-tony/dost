MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHROMA_DB_PATH = "./chroma_db"
EMBEDDER_MODEL = "all-MiniLM-L6-v2"
