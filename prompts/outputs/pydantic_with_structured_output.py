from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field, EmailStr

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    provider="fireworks-ai",        # NOT cerebras: cerebras 400s on structured output
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# schema
class Review(BaseModel):
    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(description="Return sentiment of the review either negative, positive or neutral")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list")
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer")
    email: Optional[EmailStr] = Field(default=None, description="The reviewer's email address if present in the review")

# method="json_schema" required: function_calling (the default) rejects Pydantic on ChatHuggingFace
structured_model = model.with_structured_output(Review, method="json_schema")

result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.
The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.
However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.
Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Review by humzah haiders
""")

# json_schema returns a plain dict on this stack, so wrap it back into the model
review = Review(**result)            # re-validates through Pydantic (types, EmailStr, constraints)

print(review.model_dump_json(indent=2))   # now attribute access works too: review.name, review.sentiment


# ─────────────────────────────────────────────────────────────────────────────
# CONCEPT: Structured Output with a Pydantic schema on the HuggingFace stack
# ─────────────────────────────────────────────────────────────────────────────
#
# THE GOAL
#   Force the LLM to return data shaped like the Review model (fixed fields,
#   fixed types) instead of free-form text, and get it back parsed. Same
#   with_structured_output() idea as the TypedDict file, but the schema is a
#   Pydantic BaseModel and we validate the result through it.
#
# TypedDict SCHEMA vs PYDANTIC SCHEMA
#   TypedDict:  field: Annotated[list[str], "desc"]        -> hint only, no validation
#   Pydantic:   field: list[str] = Field(description="desc") -> validates + coerces,
#               and Field() can add real constraints (gt, lt, min_length, ...).
#
# FIELD NOTES
#   key_themes / summary / sentiment       required
#   sentiment: Literal["pos","neg"]        output locked to exactly these two
#                                          (description mentions "neutral" but the
#                                          Literal forbids it — Literal wins)
#   pros / cons / name / email             Optional[...] = Field(default=None):
#                                          model may omit -> becomes None
#   email: Optional[EmailStr]              EmailStr validates format IF a value is
#                                          returned. This review has no email, so
#                                          the result is email=None.
#                                          NEEDS `pip install email-validator`, or
#                                          the file fails at class-definition time.
#   Field(description=...) is sent to the LLM as the per-field instruction;
#   Field(default=None) is what makes a field skippable.
#
# ── THE THREE PROVIDER-SPECIFIC GOTCHAS THIS FILE EXISTS TO REMEMBER ──────────
#
# 1) method="json_schema" is MANDATORY here.
#    ChatHuggingFace.with_structured_output defaults to method="function_calling",
#    and that path raises: "Pydantic schema is not supported for function calling"
#    (TypedDict works there, Pydantic doesn't — a langchain-huggingface limitation).
#    The three methods are: "function_calling", "json_schema", "json_mode".
#    json_schema is the one that accepts a Pydantic class.
#
# 2) The PROVIDER must actually honor structured output.
#    With provider="cerebras" this exact call returns HTTP 400 Bad Request —
#    Cerebras rejects LangChain's structured-output request format (won't take a
#    forced tool_choice, and forbids tools + response_format together). Cerebras
#    is fine for plain chat, but NOT for this. provider="fireworks-ai" (or
#    "together") fixes it, because those providers implement the json_schema path.
#
# 3) json_schema returns a DICT, not a Review object.
#    On this stack the json_schema path does NOT run the Pydantic parser to
#    rebuild the object (langchain issue #32197) — invoke() hands back a plain
#    dict. That's why we do `Review(**result)`: it feeds the dict back through the
#    model to re-validate and give us the actual Review object (so .model_dump_json()
#    and attribute access like review.name work). Without wrapping, you'd read it
#    as result['name'] and get no validation.
#
#    Takeaway: on the HF stack, structured output is provider-sensitive.
#      TypedDict + function_calling  -> works (most providers), returns dict
#      Pydantic  -> needs method="json_schema" -> needs a supporting provider
#                -> returns a dict -> wrap Review(**result) for a validated object
#
# ROBUST ALTERNATIVE (any provider, incl. cerebras): PydanticOutputParser
#   Instead of with_structured_output, prompt the model to emit JSON and parse it:
#       parser = PydanticOutputParser(pydantic_object=Review)
#       chain  = prompt_with(parser.get_format_instructions()) | model | parser
#   This never touches the provider's native structured-output API, so it works
#   everywhere — the go-to fallback when a provider won't cooperate. It also
#   returns a real Review object directly (no wrapping needed).
# ─────────────────────────────────────────────────────────────────────────────