from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",       # gemma-2-2b-it 400s: model_not_supported
    provider="cerebras",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person')      # gt=18 -> must be > 18 (validated)
    city: str = Field(description='Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)   # parses text -> validated Person object

template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}   # tells model the exact JSON shape
)

# ── manual version (no chain) — do each step by hand ─────────────────────────
prompt = template.invoke({'place': 'sri lankan'})   # fill template -> PromptValue
result = model.invoke(prompt)                        # LLM call -> AIMessage
final_result = parser.parse(result.content)          # parse the .content STRING -> Person object

print(final_result)                                  # a validated Person (name, age, city)


# ─────────────────────────────────────────────────────────────────────────────
# CONCEPT: PydanticOutputParser — and chain vs manual (.content) form
# ─────────────────────────────────────────────────────────────────────────────
#
# WHAT THIS FILE SHOWS
#   The SAME PydanticOutputParser flow, but written the MANUAL way instead of the
#   LCEL chain. Compare the two forms — they do identical work:
#
#     CHAIN (what you had):
#         chain = template | model | parser
#         final_result = chain.invoke({'place': 'sri lankan'})
#
#     MANUAL (this file):
#         prompt = template.invoke({'place': 'sri lankan'})   # the | template step
#         result = model.invoke(prompt)                        # the | model step
#         final_result = parser.parse(result.content)          # the | parser step
#
#   Each | in the chain is one of these lines. The pipe just auto-passes each
#   step's output to the next; here you pass it by hand.
#
# THE ONE THING TO NOTICE — parser.parse(result.content)
#   In the chain, `| parser` received the model's AIMessage and parsed it for you.
#   Doing it manually, YOU must:
#     1. pull the text out of the AIMessage -> result.content  (a string)
#     2. feed that string to parser.parse() -> a Person object
#   parser.parse() is the manual trigger for what `| parser` did automatically.
#   (Note: the chain hands the parser an AIMessage, while .parse() here takes the
#    raw string result.content — same end result, a validated Person.)
#
# HOW PydanticOutputParser WORKS (both forms)
#   get_format_instructions()  -> generates prompt text describing the EXACT JSON
#                                 shape from your Person model (fields + types).
#                                 Injected via {format_instruction} (partial_variables,
#                                 since it's constant — same pattern as the other parsers).
#   parser.parse(text)         -> reads the model's JSON text, builds a Person,
#                                 and VALIDATES it through Pydantic.
#
# WHY THIS IS THE "BEST" PARSER — it validates
#   age: Field(gt=18) is ENFORCED. If the model returns age=15, parsing RAISES a
#   ValidationError instead of silently accepting it. That's the edge over the
#   other parsers:
#     JsonOutputParser        -> valid JSON, no fixed shape, no validation
#     StructuredOutputParser  -> fixed keys, but string-only, no validation (and
#                                now legacy -> moved to langchain_classic)
#     PydanticOutputParser    -> fixed shape + types + constraints, validated     <-- this
#   And unlike with_structured_output, it's pure prompt+parse -> works on ANY
#   provider (cerebras included), no 400s.
#
# result.content — WHY .content EVERY TIME (recap)
#   model.invoke() returns an AIMessage, not a string. .content is the text field.
#   The manual form forces you to touch it directly; the chain form hides it. This
#   is exactly the .content juggling that StrOutputParser / a parser removes for
#   you in chain form.
#
# NOTE ON MODEL / PROVIDER
#   Prompt+parse only, so cerebras is fine. gemma-2-2b-it swapped out — no enabled
#   provider hosts it on your account (model_not_supported).
# ─────────────────────────────────────────────────────────────────────────────