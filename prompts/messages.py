from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

# Load environment variables from .env (needs HUGGINGFACEHUB_API_TOKEN)
load_dotenv()

# Set up the base LLM endpoint via Hugging Face's hosted API (free tier)
# - repo_id / provider: confirmed working combo (openai/gpt-oss-120b via Cerebras)
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    provider="cerebras",
    task="text-generation",
    max_new_tokens=1024  # increase this if answers get cut off
)

# Wrap the raw endpoint in a chat interface
model = ChatHuggingFace(llm=llm)

# A one-off, manually built list of messages:
# - SystemMessage: sets the assistant's behavior/persona
# - HumanMessage: the user's single question
messages = [
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me about LangChain')
]

# Send this fixed message list to the model — single call, no loop
result = model.invoke(messages)

# Manually append the model's reply back into the list
# (this shows HOW conversation history is built message-by-message,
#  but the exchange ends here — there is no next turn)
messages.append(AIMessage(content=result.content))

print(messages)

# ---------------------------------------------------------------------------
# DIFFERENCE FROM chatbot.py:
#
# This file demonstrates the CONCEPT of message-based conversation —
# how SystemMessage / HumanMessage / AIMessage fit together and how a
# conversation history list is built by hand. It runs ONCE: one question,
# one answer, then the script ends. There's no user interaction loop.
#
# chatbot.py takes this same message-list pattern and turns it into an
# ACTUAL working chatbot: it wraps the invoke() call in a while-loop that
# keeps asking for user input, keeps appending both sides of the
# conversation to chat_history, and keeps feeding the growing history back
# into the model — so the model has full conversational memory across
# multiple turns, until the user types "exit".
# ---------------------------------------------------------------------------