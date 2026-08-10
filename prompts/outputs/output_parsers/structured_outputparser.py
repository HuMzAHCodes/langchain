from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",       # gemma-2-2b-it 400s: model_not_supported
    provider="cerebras",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# define the shape: each ResponseSchema = one required field the model must return
schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)   # build parser from that shape

template = PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}  # tells model the exact keys
)

chain = template | model | parser         # fill prompt -> LLM -> parse into a fixed-key dict

result = chain.invoke({'topic': 'black hole'})

print(result)                             # dict with EXACTLY keys fact_1, fact_2, fact_3


# ─────────────────────────────────────────────────────────────────────────────
# CONCEPT: StructuredOutputParser + ResponseSchema
# ─────────────────────────────────────────────────────────────────────────────
#
# WHERE THIS SITS — the parser progression
#   StrOutputParser        -> gives you a plain string (.content)
#   JsonOutputParser       -> gives you JSON, but the SHAPE is whatever the model
#                             decided (no fixed keys)
#   StructuredOutputParser -> gives you JSON with the EXACT keys YOU declared     <-- this file
#   PydanticOutputParser   -> same fixed keys PLUS type validation (next step up)
#
#   So this is the fix for JsonOutputParser's weakness: last file, the model could
#   return a list one run and a dict the next. Here you PIN the keys: the result
#   is guaranteed to have fact_1, fact_2, fact_3.
#
# ResponseSchema — declaring each field
#   Each ResponseSchema(name=..., description=...) defines ONE output field:
#     name        -> the key that will appear in the result dict
#     description -> what should go in it (this text steers the model per field,
#                    same role as Field(description=...) in Pydantic or the
#                    Annotated string in TypedDict)
#   The list of them = the full shape you want back.
#
# from_response_schemas + get_format_instructions — the mechanism
#   StructuredOutputParser.from_response_schemas(schema) builds a parser that
#   knows your desired shape. Its get_format_instructions() then generates prompt
#   text spelling out EXACTLY which keys to return and in what JSON format. You
#   inject that via {format_instruction} (pre-filled with partial_variables,
#   since it never changes — same pattern as the JsonOutputParser file).
#   After the call, the parser reads the model's text back into a dict with those
#   keys.
#
# THE FLOW
#   {'topic': 'black hole'}
#     -> template   fills {topic}; {format_instruction} already set -> PromptValue
#     -> model      LLM call, told to return keys fact_1/2/3        -> AIMessage
#     -> parser     parse text -> {'fact_1': ..., 'fact_2': ..., 'fact_3': ...}
#
# THE KEY LIMITATION vs Pydantic — NO TYPE VALIDATION
#   StructuredOutputParser guarantees the KEYS exist, but every value comes back
#   as a STRING and nothing is type-checked or constrained. You can't say "this
#   field must be an int" or "must be > 0" or "must be a list." It's shape-only.
#   -> When you need types + validation (int, enums, ranges, nested objects),
#      step up to PydanticOutputParser. StructuredOutputParser is the lightweight
#      "just give me these named fields as strings" option.
#
# WHY THIS WORKS ON CEREBRAS (when with_structured_output didn't)
#   Like the other parsers, this is pure PROMPT + PARSE — it never calls the
#   provider's native structured-output API. So it runs on ANY provider,
#   cerebras included. That's the recurring theme: parsers = portable,
#   with_structured_output = provider-dependent.
#
# NOTE ON THE IMPORT
#   StructuredOutputParser and ResponseSchema come from `langchain.output_parsers`
#   (the main langchain package), NOT langchain_core like the other parsers.
#   If you hit ModuleNotFoundError, make sure `langchain` itself is installed
#   (pip install langchain), not just langchain-core.
# ─────────────────────────────────────────────────────────────────────────────