from configuration import *

# ============= MODEL SETUP ============= #
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

# --- LLM (Ollama local model) --- #
llm = Ollama(model="llama3.2:3b")

# ========== EMBEDDINGS ========== #
embedding_model = HuggingFaceEmbeddings(
    model_name=embedding_model_name
)