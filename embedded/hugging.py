from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env (needs HUGGINGFACEHUB_API_TOKEN)
load_dotenv()


embedding = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

# Sample documents to embed
documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

# Generate embeddings (vector representations) for each document via the API
vector = embedding.embed_documents(documents)

print(str(vector))