from dotenv import load_dotenv
  



from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(dotenv_path=r"D:\Gen_AI\LangChain_Models\.env")

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    provider="cerebras",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

chain = prompt | model | parser         # the whole flow as one pipeline

result = chain.invoke({'topic': 'cricket'})
print(result)

chain.get_graph().print_ascii()         # draws the chain's structure as ASCII


# ─────────────────────────────────────────────────────────────────────────────
# CONCEPT: The simplest chain + visualizing it with get_graph().print_ascii()
# ─────────────────────────────────────────────────────────────────────────────
#
# THE CHAIN (nothing new — you've built this before)
#   prompt | model | parser  is a basic SEQUENTIAL chain:
#     {'topic': 'cricket'}
#       -> prompt   fills {topic}         -> PromptValue
#       -> model    LLM call              -> AIMessage
#       -> parser   extract .content      -> plain string
#   chain.invoke(...) runs all three left to right and returns the final string.
#   This is the foundation the chains topic builds on (sequential -> parallel ->
#   conditional all start from this | pipe).
#
# THE NEW BIT: chain.get_graph().print_ascii()
#   Every LCEL chain is internally a GRAPH of runnables. get_graph() returns that
#   structure; .print_ascii() draws it in the terminal so you can SEE the wiring:
#
#       +-------------+
#       | PromptInput |          <- the input that enters the chain
#       +-------------+
#              *
#       +----------------+
#       | PromptTemplate |       <- step 1
#       +----------------+
#              *
#       +-----------------+
#       | ChatHuggingFace |      <- step 2 (the model)
#       +-----------------+
#              *
#       +-----------------+
#       | StrOutputParser |      <- step 3
#       +-----------------+
#              *
#       +-----------------------+
#       | StrOutputParserOutput |  <- the final output leaving the chain
#       +-----------------------+
#
#   (Exact box names vary; a straight line = a simple sequential chain.)
#
# WHY THIS MATTERS LATER
#   Right now the graph is a boring straight line — but that's the point of
#   learning it here on the simplest chain. Once you hit PARALLEL chains
#   (RunnableParallel) and CONDITIONAL/ROUTER chains, print_ascii() shows the
#   branches and merges visually, which makes debugging far easier than reading
#   the code. Learn the tool on the trivial case now; lean on it when the graph
#   actually forks.
#
# NOTE ON MODEL / PROVIDER (the usual)
#   Plain prompt+parse, no structured output -> cerebras is fine. Using the
#   known-good gpt-oss-120b + cerebras combo.
# ─────────────────────────────────────────────────────────────────────────────