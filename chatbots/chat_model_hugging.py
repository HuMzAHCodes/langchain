from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

# Load environment variables from .env (needs HUGGINGFACEHUB_API_TOKEN)
load_dotenv()

# ---------------------------------------------------------------------------
# MODEL HISTORY / REASONING
#
# 1st model tried: TinyLlama/TinyLlama-1.1B-Chat-v1.0
#   -> Failed with error: model_not_supported
#   -> Reason: Hugging Face's Inference Providers router only serves models
#      that a partner provider (Cerebras, Together, Fireworks, DeepInfra, etc.)
#      has actually chosen to host. TinyLlama, despite being a well-known
#      open-source model, isn't hosted by any of them — so the router
#      rejected it regardless of API key or code correctness.
#
# 2nd model tried: meta-llama/Llama-3.2-3B-Instruct
#   -> Hit the exact same model_not_supported error.
#   -> Confirmed this wasn't a one-off issue — many popular repo_ids simply
#      aren't wired up to any router provider, so guessing "well-known"
#      model names wasn't a reliable strategy.
#
# How the working model was found:
#   -> Instead of guessing again, tested directly against Hugging Face's
#      own documented working example (openai/gpt-oss-120b:cerebras) using
#      the raw huggingface_hub InferenceClient. That call succeeded,
#      confirming both the model and provider were live.
#
# Current model: openai/gpt-oss-120b, provider="cerebras"
#   -> Explicitly pinning the provider (instead of leaving it on auto-route)
#      avoids ambiguous routing failures and guarantees the request goes to
#      a provider that actually serves this model — which is why it works
#      reliably here.
# ---------------------------------------------------------------------------

# Set up the base LLM endpoint
# - repo_id: the model to call
# - provider: pins the request to a specific inference provider (Cerebras)
# - task: type of generation task
# - temperature: controls randomness (higher = more creative/random output)
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    provider="cerebras",
    task="text-generation",
    temperature=1.2
)

# Wrap the raw LLM endpoint in a chat interface
# (adds proper chat-style prompting/formatting on top of the base model)
model = ChatHuggingFace(llm=llm)

# Send the prompt and get a response
result = model.invoke("generate a 60 lines poem on cricket")

# Print just the generated text content
print(result.content)