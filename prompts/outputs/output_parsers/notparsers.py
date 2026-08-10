from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

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
    template='Write a 5 line summary on the following text. \n {text}',   # \n not /n — see note
    input_variables=['text']
)

prompt1 = template1.invoke({'topic': 'black hole'})   # fill template1 -> a PromptValue
result = model.invoke(prompt1)                         # LLM call #1 -> AIMessage (the report)

prompt2 = template2.invoke({'text': result.content})   # feed report TEXT into template2
result1 = model.invoke(prompt2)                        # LLM call #2 -> AIMessage (the summary)

print(result1.content)                                 # .content = the string; the message also carries metadata


# ─────────────────────────────────────────────────────────────────────────────
# CONCEPT: Sequential Prompt Chaining (done manually)
# ─────────────────────────────────────────────────────────────────────────────
#
# THE IDEA
#   Two LLM calls wired end to end: the OUTPUT of the first becomes the INPUT of
#   the second. Here: generate a detailed report on "black hole", then feed that
#   report into a second prompt that summarizes it in 5 lines. This is a "chain"
#   — multiple steps where each depends on the previous one's result.
#
# THE FLOW, STEP BY STEP
#   template1.invoke({'topic': 'black hole'})
#       Fills the {topic} placeholder -> produces a PromptValue (a ready-to-send
#       prompt), NOT a string and NOT an LLM answer yet.
#   model.invoke(prompt1) -> result
#       Sends it to the LLM. Returns an AIMessage. The actual text lives in
#       result.content (that's why every model call here reads .content).
#   template2.invoke({'text': result.content})
#       THE HANDOFF: the first answer's text is injected into the second template
#       as {text}. This link is what makes it a chain rather than two unrelated
#       calls.
#   model.invoke(prompt2) -> result1
#       Second LLM call. result1.content is the final 5-line summary.
#
# WHY .content EVERY TIME
#   model.invoke() returns an AIMessage object, not a bare string. To pass the
#   text onward or print it, you take .content. (The object also holds metadata
#   like token usage / response info — the string is just one part of it.)
#
# THE BUG I FIXED: '/n' -> '\n'
#   Your template2 had "/n" (forward slash + n), which is just the literal
#   characters "/n" — it does NOT create a newline. The escape for a newline is
#   "\n" (backslash + n). With the old version the model still worked but the
#   text ran on without the intended line break; "\n" puts the review text on its
#   own line, which is what you wanted.
#
# THIS IS THE "MANUAL" VERSION — LCEL DOES IT IN ONE LINE
#   Notice the repetitive pattern: invoke a template, call the model, pull
#   .content, invoke the next template, call the model again. LangChain's LCEL
#   (the | pipe syntax) collapses all of this into a single declarative chain:
#
#       from langchain_core.output_parsers import StrOutputParser
#       parser = StrOutputParser()          # auto-extracts .content for you
#       chain = template1 | model | parser | template2 | model | parser
#       final = chain.invoke({'topic': 'black hole'})
#
#   Same two-step report->summary flow, but the pipe passes each step's output to
#   the next automatically, and StrOutputParser saves you from writing .content
#   over and over. Seeing the manual version FIRST is the point — it shows you
#   exactly what the | operator is doing under the hood before the syntax hides it.
#
# NOTE ON THIS MODEL / PROVIDER
#   repo_id="google/gemma-2-2b-it" with NO provider set -> HF auto-routes to
#   whichever provider serves it. This is a plain chat task (no structured
#   output), so it doesn't hit the cerebras/fireworks structured-output issue
#   from the earlier files. If you ever get model_not_supported, the fix is the
#   same as before: pick a repo_id + provider combo a partner actually hosts.
# ─────────────────────────────────────────────────────────────────────────────