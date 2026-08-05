"""
NOTE: This script demonstrates running a Hugging Face model LOCALLY
(fully downloaded to disk and executed on this machine), as opposed
to calling a hosted API.

This file is kept for reference/learning purposes to show the concept
and code pattern for local inference. It has NOT been run in this repo —
TinyLlama-1.1B's weights (~2.2GB download) plus PyTorch's footprint were
too large for available disk space/bandwidth at the time. The API-based
approach (see chat_model_hugging.py) was used instead for actually
generating output.

Requirements to actually run this:
    pip install torch
    (first run downloads ~2.2GB of model weights into HF_HOME)
"""

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

# Set where Hugging Face caches downloaded models/weights
# Redirected to D: drive to avoid filling up the default (usually C:) cache location
os.environ['HF_HOME'] = 'D:/huggingface_cache'

# Load the model to run LOCALLY on this machine (not via HF's hosted API)
# - model_id: which model to download and run
# - task: type of generation task
# - pipeline_kwargs: generation settings passed to the underlying HF pipeline
#     - temperature: controls randomness (lower = more focused/deterministic)
#     - max_new_tokens: caps how many tokens the model generates in response
llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)

# Wrap the local pipeline in a chat interface
# (adds chat-style prompting/formatting on top of the raw local model)
model = ChatHuggingFace(llm=llm)

# Run inference locally — no internet/API call needed after the model is downloaded
result = model.invoke("What is the capital of India")

print(result.content)