from langchain_core.prompts import ChatPromptTemplate

# A ChatPromptTemplate lets you define a reusable, multi-message prompt
# structure with placeholders (like {domain}, {topic}) filled in later.
# Each tuple is (role, message_text):
#   - 'system': sets the assistant's behavior/persona
#   - 'human':  the user's message
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain in simple terms, what is {topic}')
])

# Fill in the placeholders with actual values.
# This does NOT call any model — it just renders the final message list
# that would be sent to a model if you chained it with one.
prompt = chat_template.invoke({'domain': 'cricket', 'topic': 'Dusra'})

print(prompt)

# ---------------------------------------------------------------------------
# CONCEPT: static prompt templating
#
# This file shows the basic building block — a fixed-shape prompt with
# variable slots ({domain}, {topic}) filled in at invoke time. It always
# produces the exact same message structure: one system message, one
# human message. There's no room here for a growing conversation history.
# ---------------------------------------------------------------------------