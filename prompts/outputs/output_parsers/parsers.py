from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    provider="cerebras",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. \n {text}',   # \n not /n
    input_variables=['text']
)

parser = StrOutputParser()                                    # pulls .content out as a plain string

chain = template1 | model | parser | template2 | model | parser   # the whole flow as one pipeline

result = chain.invoke({'topic': 'black hole'})                # run it end to end -> final summary string

print(result)


# ─────────────────────────────────────────────────────────────────────────────
# CONCEPT: LCEL Chains (the | pipe) + StrOutputParser
# ─────────────────────────────────────────────────────────────────────────────
#
# SAME FLOW AS THE MANUAL FILE — collapsed
#   Previous file did this by hand:
#       prompt1 = template1.invoke(...); result = model.invoke(prompt1)
#       prompt2 = template2.invoke({'text': result.content}); ...
#   This file does the identical report->summary chain in ONE line with |.
#   Nothing new conceptually — LCEL just automates the wiring you did manually.
#
# WHAT THE | (PIPE) DOES
#   a | b means "run a, feed its output as b's input." LangChain calls each piece
#   a Runnable, and | composes them left to right. So:
#       template1 | model | parser | template2 | model | parser
#   flows as:
#       {'topic': 'black hole'}
#         -> template1   fills {topic}        -> PromptValue
#         -> model       LLM call #1          -> AIMessage (the report)
#         -> parser      extract .content     -> report as a STRING
#         -> template2   fills {text}         -> PromptValue (summary prompt)
#         -> model       LLM call #2          -> AIMessage (the summary)
#         -> parser      extract .content     -> final summary STRING
#   The output of each stage is automatically handed to the next — no temp
#   variables, no manual .content juggling.
#
# WHY StrOutputParser IS NEEDED BETWEEN model AND template2
#   model outputs an AIMessage, but template2 expects a plain string for {text}.
#   StrOutputParser sits in between and extracts .content -> string. That's the
#   whole job it does here: it replaces the `result.content` you wrote by hand in
#   the manual version. The final parser does the same so `result` is a clean
#   string, not an AIMessage.
#
# HOW THE INPUT DICT FLOWS
#   chain.invoke({'topic': 'black hole'}) — the dict goes to the FIRST runnable
#   (template1), which needs 'topic'. The second template needs 'text', but you
#   DON'T supply that — it's produced mid-chain by the first model+parser and
#   piped in automatically. You only ever provide the chain's starting input.
#
# WHY LCEL INSTEAD OF MANUAL invoke() CALLS
#   - Less boilerplate: no intermediate variables, no repeated .content.
#   - Free features: the same chain also supports .stream() (token-by-token),
#     .batch() (many inputs at once), and async — without rewriting it.
#   - Readable: the pipeline reads like the data flow itself.
#
# NOTE ON MODEL / PROVIDER
#   Plain chat (no structured output), so cerebras is fine here — the
#   cerebras-400 issue only affects with_structured_output. Using the known-good
#   gpt-oss-120b + cerebras combo (gemma-2-2b-it failed earlier with
#   model_not_supported — no provider hosts it on your account).
# ─────────────────────────────────────────────────────────────────────────────