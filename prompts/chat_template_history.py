from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Same idea as chat_template_basic.py, but this time we add a
# MessagesPlaceholder — a slot that isn't a fixed string, but instead
# gets filled with a LIST of past messages at invoke time. This is what
# lets a prompt template include a variable-length conversation history
# instead of just one fixed human message.
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

# Start with an empty history list
chat_history = []

# Load previously saved conversation lines from a text file
# (this is a simple/manual way of persisting history across runs —
#  contrast with chatbot.py's chat_history, which only lived in memory
#  and vanished when the script ended)
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)

# Fill the template: chat_history goes into the MessagesPlaceholder slot,
# query becomes the final human message
prompt = chat_template.invoke({'chat_history': chat_history, 'query': 'Where is my refund'})

print(prompt)

# ---------------------------------------------------------------------------
# RELATION TO chat_template_basic.py:
#
# chat_template_basic.py builds a prompt with a FIXED number of messages
# (one system + one human) — good for single-turn, stateless prompts.
#
# chat_template_history.py extends that same ChatPromptTemplate concept
# using MessagesPlaceholder, which allows an ARBITRARY number of prior
# messages (chat_history) to be inserted into the template before the
# final human query. This is the templating-side counterpart to what
# chatbot.py did manually with a Python list — instead of hand-building
# the message list every loop iteration, ChatPromptTemplate lets you
# declare "history goes here" once, and plug in whatever history you have
# (freshly loaded from a file, a database, memory, etc.) at invoke time.
# ---------------------------------------------------------------------------