from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",       # gemma-2-2b-it 400s: model_not_supported
    provider="cerebras",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()               # parses the model's text into a Python dict/list

template = PromptTemplate(
    template='Give me 5 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],            # filled at invoke time
    partial_variables={'format_instruction': parser.get_format_instructions()}  # filled NOW, once
)

chain = template | model | parser         # fill prompt -> LLM -> parse text into JSON

result = chain.invoke({'topic': 'black hole'})

print(result)                             # a Python dict/list (already parsed), not a raw string


# ─────────────────────────────────────────────────────────────────────────────
# CONCEPT: JsonOutputParser + partial_variables / format_instructions
# ─────────────────────────────────────────────────────────────────────────────
#
# WHAT JsonOutputParser DOES
#   StrOutputParser (last file) pulled the plain .content string out of the
#   AIMessage. JsonOutputParser goes further: it takes the model's text, expects
#   it to be JSON, and PARSES it into a real Python object (dict or list). So
#   `result` here is already a dict you can index — not a string you'd have to
#   json.loads() yourself.
#
# HOW THE MODEL KNOWS TO RETURN JSON — format_instructions
#   A parser can only parse JSON if the model actually PRODUCED JSON. That's the
#   job of parser.get_format_instructions(): it returns a chunk of text telling
#   the model "return your answer as JSON" (with formatting rules). You inject
#   that text into the prompt via {format_instruction}. So the parser does two
#   things across the chain:
#     - BEFORE the call: supplies instructions that steer the model to emit JSON
#     - AFTER the call:  parses that JSON back into a Python object
#
# WHY partial_variables (this is the new PromptTemplate feature)
#   The template has TWO placeholders: {topic} and {format_instruction}.
#     - {topic}              changes every call    -> listed in input_variables,
#                                                     supplied at chain.invoke()
#     - {format_instruction} is always the SAME    -> filled ONCE, up front, via
#                                                     partial_variables
#   partial_variables = "pre-fill this placeholder now so the caller doesn't have
#   to pass it every time." get_format_instructions() never changes, so it's a
#   perfect fit. That's why chain.invoke({'topic': ...}) only passes topic —
#   format_instruction is already baked in.
#
# THE FLOW
#   {'topic': 'black hole'}
#     -> template   fills {topic}, {format_instruction} already set -> PromptValue
#     -> model      LLM call, guided to output JSON                 -> AIMessage
#     -> parser     reads the JSON text -> Python dict/list         -> result
#
# JsonOutputParser vs with_structured_output — important distinction
#   Both get you structured data, but differently:
#     with_structured_output -> uses the PROVIDER'S native json_schema/tool API;
#                               provider-sensitive (cerebras 400s, needs fireworks)
#     JsonOutputParser       -> pure PROMPT + PARSE; works on ANY provider,
#                               cerebras included, because it's just text in/out
#   BUT the trade-off: JsonOutputParser does NOT enforce a schema. It parses
#   whatever JSON shape the model returns — you have no guarantee of specific keys
#   or types. It only guarantees "valid JSON," not "JSON matching MY structure."
#   -> If you need a GUARANTEED shape, use PydanticOutputParser (next step up):
#      same prompt+parse approach, but validated against a Pydantic model.
#
# NOTE ON MODEL / PROVIDER
#   Plain prompt+parse, so cerebras is fine (no native structured-output call).
#   gemma-2-2b-it was swapped out — no enabled provider hosts it on your account.
# ─────────────────────────────────────────────────────────────────────────────