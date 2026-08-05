from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables from .env (needs HUGGINGFACEHUB_API_TOKEN)
load_dotenv()

# Set up embedding model via Hugging Face's hosted API (free tier)
# - model: sentence-transformers model used to convert text into vectors
# - task: required for embedding models on the router
embedding = HuggingFaceEndpointEmbeddings(
    model='sentence-transformers/all-MiniLM-L6-v2',
    task='feature-extraction'
)

# Sample documents to search over
documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

# The search query we want to match against the documents
query = 'tell me about bumrah'

# Convert all documents into embedding vectors
doc_embeddings = embedding.embed_documents(documents)

# Convert the query into its own embedding vector
query_embedding = embedding.embed_query(query)

# Compare the query vector against every document vector using cosine similarity
# (measures how close the vectors are in direction, i.e. semantic similarity)
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# Sort (index, score) pairs by score ascending, then take the last one
# -> gives the index and score of the MOST similar document
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(query)
print(documents[index])
print("similarity score is:", score)