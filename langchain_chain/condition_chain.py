from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv(dotenv_path=r"D:\Gen_AI\LangChain_Models\.env")

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    provider="cerebras",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)   # forces output into a validated Feedback object

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2       # step 1: feedback -> Feedback(sentiment=...)

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)
prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser),   # (condition, chain-to-run-if-true)
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")               # default / fallback (no condition)
)

chain = classifier_chain | branch_chain            # classify, THEN pick a branch based on the result

print(chain.invoke({'feedback': 'This is a beautiful phone'}))

chain.get_graph().print_ascii()


# ─────────────────────────────────────────────────────────────────────────────
# CONCEPT: Conditional Chains — RunnableBranch + RunnableLambda
# ─────────────────────────────────────────────────────────────────────────────
#
# THE IDEA — routing / if-else for chains
#   So far chains were straight (sequential) or forked-and-merged (parallel).
#   This one is CONDITIONAL: it runs a DIFFERENT downstream chain depending on an
#   earlier result. Here: classify feedback -> if positive, write a positive
#   reply; if negative, write a negative reply. It's an if/elif/else built out of
#   runnables.
#
# THE TWO-STAGE FLOW
#   chain = classifier_chain | branch_chain
#     classifier_chain: prompt1 | model | parser2
#         feedback text -> LLM -> PydanticOutputParser -> Feedback(sentiment='positive')
#         (a validated object, so the next stage can read x.sentiment safely)
#     branch_chain: looks at that Feedback object and routes to one sub-chain.
#
# RunnableBranch — the router
#   Signature: RunnableBranch( (cond1, chain1), (cond2, chain2), ..., default )
#     - Each tuple is (CONDITION function, CHAIN to run if the condition is True).
#     - Conditions are checked TOP-DOWN; the FIRST one that returns True wins and
#       its chain runs. The rest are skipped.
#     - The LAST argument has NO condition — it's the DEFAULT, run when none of the
#       conditions matched (the else branch).
#   Here:
#     (lambda x: x.sentiment == 'positive', prompt2 | model | parser)  -> positive reply
#     (lambda x: x.sentiment == 'negative', prompt3 | model | parser)  -> negative reply
#     RunnableLambda(lambda x: "could not find sentiment")             -> fallback
#
# WHAT `x` IS INSIDE THE BRANCH
#   x is whatever classifier_chain output = a Feedback OBJECT. That's why the
#   conditions read x.sentiment (attribute access). If parser2 had returned a dict
#   instead, you'd write x['sentiment']. This is exactly why the classifier ends
#   in PydanticOutputParser — it guarantees a typed object with a .sentiment field
#   for the router to test.
#
# RunnableLambda — wrap a plain function as a runnable
#   A chain step must be a "runnable" so | and RunnableBranch can call it. A raw
#   python function isn't one. RunnableLambda(fn) wraps any function so it fits in
#   a chain. Used here for the fallback: it just returns a fixed string. You can
#   also use RunnableLambda anywhere you need custom logic (reshape data, post-
#   process) mid-chain.
#   (Note: the two branch conditions are plain lambdas — as the FIRST item of a
#    tuple they don't need wrapping. The fallback is a standalone step, so it DOES
#    need RunnableLambda to be a valid runnable.)
#
# WHY the sentiment field is Literal['positive','negative']
#   Constrains the classifier's output to exactly those two values, so the branch
#   conditions are guaranteed to match one of them (or fall through to default).
#   Without the constraint the model might return "Positive" / "mixed" / etc. and
#   silently hit the fallback.
#
# get_graph().print_ascii()
#   Shows the branch visually: a straight classifier section, then the graph SPLITS
#   into the possible routes and rejoins — the conditional structure made visible.
#
# NOTE ON MODEL / PROVIDER
#   The classifier uses PydanticOutputParser (prompt+parse), NOT
#   with_structured_output — so cerebras is fine here; this is the portable path
#   that dodges the provider structured-output 400. One model reused everywhere.
#
# IMPORT FIX
#   RunnableBranch/RunnableParallel/RunnableLambda now live in
#   langchain_core.runnables — the old `langchain.schema.runnable` path throws
#   ModuleNotFoundError on langchain 1.x.
# ─────────────────────────────────────────────────────────────────────────────