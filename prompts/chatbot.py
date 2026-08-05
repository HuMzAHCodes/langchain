from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

# Load environment variables from .env (needs HUGGINGFACEHUB_API_TOKEN)
load_dotenv()

# Set up the base LLM endpoint via Hugging Face's hosted API (free tier)
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    provider="cerebras",
    task="text-generation"
)

# Wrap the raw endpoint in a chat interface
model = ChatHuggingFace(llm=llm)

# Start the conversation history with a system message to set behavior.
# This list will keep growing as the conversation continues.
chat_history = [
    SystemMessage(content='You are a helpful AI assistant')
]

# Keep the conversation going until the user types "exit"
while True:
    user_input = input('You: ')

    # Add the user's message to the ongoing history
    chat_history.append(HumanMessage(content=user_input))

    if user_input == 'exit':
        break

    # Send the FULL history (not just the latest message) to the model —
    # this is what gives the model memory of earlier turns in the conversation
    result = model.invoke(chat_history)

    # Add the model's reply to the history too, so it's included in the
    # next loop's invoke() call
    chat_history.append(AIMessage(content=result.content))

    print("AI: ", result.content)

print(chat_history)

# ---------------------------------------------------------------------------
# DIFFERENCE FROM messages.py:
#
# messages.py is a static, single-turn demo — it builds one message list
# by hand, sends it once, appends the reply, and ends. It exists to show
# the basic building blocks (SystemMessage / HumanMessage / AIMessage).
#
# chatbot.py takes those same building blocks and makes them INTERACTIVE
# and MULTI-TURN: a while-loop repeatedly takes user input, appends it to
# chat_history, sends the entire growing history to the model each time
# (giving it memory of past turns), appends the AI's reply, and repeats —
# until the user exits. This is what makes it an actual usable chatbot
# rather than a one-shot message example.
# ---------------------------------------------------------------------------