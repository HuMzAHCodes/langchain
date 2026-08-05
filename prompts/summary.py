"""
=============================================================================
SESSION SUMMARY: Ways of calling .invoke() on a chat model in LangChain
=============================================================================

This session covered two broad categories of invoking a model:
    1. Single message invocation (static and dynamic)
    2. List-of-messages invocation (manual message objects, and via
       ChatPromptTemplate)

Each is explained below with its use case and underlying logic.
-----------------------------------------------------------------------------


-----------------------------------------------------------------------------
1. SINGLE MESSAGE — STATIC
-----------------------------------------------------------------------------
Example:
    result = model.invoke("What is the capital of Iran")

Logic:
    - You pass a plain string directly to invoke().
    - LangChain wraps it internally as a single HumanMessage before
      sending it to the model.
    - No system instructions, no history — just one question, one answer.

Use case:
    - Quick, one-off queries where no persona/behavior control or
      conversation context is needed.
    - Good for testing a model connection (which is exactly how we used
      it early in this session — verifying OpenAI/Gemini/HF setups worked).


-----------------------------------------------------------------------------
2. SINGLE MESSAGE — DYNAMIC (via PromptTemplate / ChatPromptTemplate)
-----------------------------------------------------------------------------
Example:
    chat_template = ChatPromptTemplate([
        ('system', 'You are a helpful {domain} expert'),
        ('human', 'Explain in simple terms, what is {topic}')
    ])
    prompt = chat_template.invoke({'domain': 'cricket', 'topic': 'Dusra'})

Logic:
    - Instead of hardcoding the text, the message contains placeholders
      ({domain}, {topic}) filled in at invoke time with real values.
    - Still produces a FIXED shape: exactly one system + one human message
      per call — only the CONTENT is dynamic, not the structure.

Use case:
    - Reusable prompt structures where the wording stays the same but
      the subject changes each time (e.g. the Research Tool app: same
      instructions, different paper/style/length selected via Streamlit
      dropdowns each run).
    - Keeps prompt design centralized and consistent instead of
      string-formatting by hand everywhere in the code.


-----------------------------------------------------------------------------
3. LIST OF MESSAGES — STATIC (SystemMessage / HumanMessage / AIMessage)
-----------------------------------------------------------------------------
Example:
    messages = [
        SystemMessage(content='You are a helpful assistant'),
        HumanMessage(content='Tell me about LangChain')
    ]
    result = model.invoke(messages)
    messages.append(AIMessage(content=result.content))

Logic:
    - A conversation is represented as an explicit Python list of typed
      message objects (System / Human / AI), each tagging WHO said what.
    - invoke() is called with the WHOLE list, not just one message — this
      is what gives the model conversational context.
    - Building/growing this list by hand is what we did in messages.py
      (single exchange) and then looped in chatbot.py (multi-turn).

Use case:
    - Manually controlled conversations where you want to see/manage
      exactly what's in the message history at every step.
    - Foundation for building an interactive chatbot: wrap this pattern
      in a while-loop, keep appending Human/AI turns, and the model
      gains memory of the whole conversation (as long as the script is
      still running — this history is NOT saved once the process ends,
      unless explicitly persisted to a file/database).


-----------------------------------------------------------------------------
4. LIST OF MESSAGES — VIA CHAT PROMPT TEMPLATE (MessagesPlaceholder)
-----------------------------------------------------------------------------
Example:
    chat_template = ChatPromptTemplate([
        ('system', 'You are a helpful customer support agent'),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', '{query}')
    ])
    prompt = chat_template.invoke({'chat_history': chat_history, 'query': 'Where is my refund'})

Logic:
    - Combines concepts 2 and 3: a templated prompt structure (system +
      human) PLUS a slot (MessagesPlaceholder) that accepts a
      variable-length list of prior messages.
    - Instead of manually rebuilding the message list every time (like
      in chatbot.py), you declare "history goes here" once in the
      template, then plug in whatever history you have — loaded from a
      file, database, or memory — at invoke time.

Use case:
    - Production-style conversational apps (e.g. customer support bots)
      where history needs to be loaded from persistent storage (a file
      or database) rather than kept only in a running script's memory.
    - Cleaner separation: the template defines STRUCTURE, the loaded
      chat_history supplies CONTENT.


-----------------------------------------------------------------------------
OVERALL PROGRESSION OF THE SESSION
-----------------------------------------------------------------------------
    Single static message
        -> Single dynamic (templated) message
            -> Manually managed list of messages (single exchange)
                -> Manually managed list of messages (looped, interactive chatbot)
                    -> Templated list of messages with a history placeholder
                       (structured, persistable conversation history)

Each step adds a layer of flexibility: from "ask one thing" to
"maintain and reuse a full, structured conversation."
=============================================================================
"""




# ---------------------------------------------------------------------------
# ADDENDUM: Hardcoded values vs. real-world user input
# ---------------------------------------------------------------------------
#
# In every example above, the values passed into invoke() were HARDCODED
# for demonstration purposes:
#     chat_template.invoke({'domain': 'cricket', 'topic': 'Dusra'})
#     chat_template.invoke({'chat_history': chat_history, 'query': 'Where is my refund'})
#
# This was done deliberately, to isolate and prove the templating mechanism
# itself works — without the added complexity of wiring up real input
# sources at the same time.
#
# In a REAL application, these values would NOT be hardcoded. They'd come
# from wherever the user actually interacts with the app:
#
#   - UI input (e.g. Streamlit):
#       st.selectbox() / st.text_input() capture user choices, and THOSE
#       variables get passed into invoke({...}) instead of fixed strings.
#       This is exactly what prompt_ui.py already does:
#           result = chain.invoke({
#               'paper_input': paper_input,   # from st.selectbox
#               'style_input': style_input,   # from st.selectbox
#               'length_input': length_input  # from st.selectbox
#           })
#
#   - Terminal/chat input (e.g. chatbot.py):
#       query = input('You: ')   # live user input each loop iteration
#
#   - chat_history in production:
#       Rather than a hardcoded chat_history.txt, a real app would pull
#       this live from a database or session store — keyed to that
#       specific user/conversation — and update it after every turn.
#
# Takeaway: the TEMPLATE STRUCTURE stays the same whether the values are
# hardcoded or dynamic. Hardcoding is just a stand-in used while learning
# or testing; production apps swap those fixed values for live data
# sourced from a UI, API request, or database.
# ---------------------------------------------------------------------------