from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    provider="cerebras",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# schema
class Review(TypedDict):
    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either negative, positive or neutral"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    name: Annotated[Optional[str], "Write the name of the reviewer"]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.
The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.
However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.
Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Review by lana del ray
""")

print(result['name'])
print(result)


# ─────────────────────────────────────────────────────────────────────────────
# CONCEPT: Structured Output with `with_structured_output()`
# ─────────────────────────────────────────────────────────────────────────────
#
# THE PROBLEM IT SOLVES
#   A raw LLM returns free-form text. If you need the answer as data your program
#   can use (a dict, specific fields, fixed value sets), parsing text by hand is
#   fragile. `with_structured_output(schema)` forces the model to return output
#   that conforms to a schema, and LangChain hands it back already parsed.
#
#   model.invoke(...)             -> AIMessage (has .content string)
#   structured_model.invoke(...)  -> a Review dict (matches the schema)
#
# HOW IT WORKS UNDER THE HOOD
#   `with_structured_output` wraps the model. Depending on the provider it uses
#   one of two mechanisms:
#     - tool / function calling: the schema is passed as a "tool" the model must
#       call; the model fills the tool's arguments, which become your fields.
#     - JSON mode: the model is told to emit JSON matching a JSON Schema.
#   gpt-oss-120b (via Cerebras) supports tool calling, so LangChain uses that by
#   default. LangChain then validates/parses the model's arguments and returns a
#   Python object. You never touch the raw text.
#
# THE SCHEMA — using a TypedDict
#   `Review` is a TypedDict: it describes the SHAPE of a dictionary (which keys
#   exist and each key's type). It is NOT a runtime class you instantiate — the
#   result is a plain dict, which is why you access it as result['name'], not
#   result.name. (Other schema styles LangChain accepts: Pydantic models -> gives
#   you an object with .name attribute access + validation; or a raw JSON Schema
#   dict. TypedDict is the lightest of the three.)
#
# THE typing PIECES, ANGLE BY ANGLE
#   Annotated[T, "..."]
#       Attaches metadata to a type. Here the T is the real type (e.g. str) and
#       the string is a DESCRIPTION. That description is sent to the model as the
#       instruction for that field — it's how you steer what goes in each slot.
#       Think of it as a per-field prompt.
#
#   list[str]
#       The field must be a list of strings (key_themes, pros, cons).
#
#   Literal["pos", "neg"]
#       Constrains the value to EXACTLY one of the listed options. The model
#       cannot return anything else for `sentiment`. NOTE: the description text
#       says "positive, negative or neutral", but the Literal only allows
#       "pos"/"neg" — the Literal is what actually enforces the output, so
#       "neutral" is impossible here. To allow it you'd write
#       Literal["pos", "neg", "neutral"]. (The Literal wins over the prose.)
#
#   Optional[T]   (same as Union[T, None])
#       The field MAY be absent / null. pros, cons, and name are optional, so if
#       the review has no cons the model can leave it out or return None without
#       breaking the schema. Required fields (key_themes, summary, sentiment)
#       must always be produced.
#
# WHY THIS MATTERS
#   You get reliable, typed, machine-usable output from an LLM — the foundation
#   for feeding model results into the rest of an app (DB rows, API responses,
#   downstream logic) instead of eyeballing a paragraph of text.
#
# HF CAVEAT (specific to this stack)
#   with_structured_output relies on the underlying model supporting tool calling
#   or JSON mode. gpt-oss-120b does. If you ever swap in a model whose HF provider
#   doesn't expose tool calling, this call can fail or return None — in that case
#   pass method="json_mode" (if supported) or fall back to a manual JSON prompt +
#   PydanticOutputParser.
# ─────────────────────────────────────────────────────────────────────────────