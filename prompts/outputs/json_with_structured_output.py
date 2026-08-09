from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    provider="fireworks-ai",        # NOT cerebras: cerebras 400s on structured output
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# schema — raw JSON Schema (a plain dict), not TypedDict or Pydantic
json_schema = {
  "title": "Review",
  "type": "object",                          # top-level shape is an object (key-value record)
  "properties": {                            # each field maps to its OWN schema object
    "key_themes": {
      "type": "array",
      "items": {"type": "string"},           # array of strings; "items" types the elements
      "description": "Write down all the key themes discussed in the review in a list"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],                # enum = the JSON Schema equivalent of Literal
      "description": "Return sentiment of the review either negative, positive or neutral"
    },
    "pros": {
      "type": ["array", "null"],             # ["array","null"] = optional/nullable (like Optional)
      "items": {"type": "string"},
      "description": "Write down all the pros inside a list"
    },
    "cons": {
      "type": ["array", "null"],
      "items": {"type": "string"},
      "description": "Write down all the cons inside a list"
    },
    "name": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]   # only these must appear; the rest are optional
}

structured_model = model.with_structured_output(json_schema)   # no method= needed: a dict schema goes straight to json_schema mode

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

print(result)                        # a plain DICT -> access as result['name'], result['sentiment']


# ─────────────────────────────────────────────────────────────────────────────
# CONCEPT: Structured Output with a RAW JSON SCHEMA (the third schema style)
# ─────────────────────────────────────────────────────────────────────────────
#
# WHERE THIS FITS — the three schema styles, now complete
#   1. TypedDict    -> Annotated[type, "desc"]     -> returns a dict
#   2. Pydantic     -> Field(description="desc")    -> returns a dict on this stack;
#                                                      wrap Review(**result) to validate
#   3. JSON Schema  -> THIS raw dict                -> returns a dict, no validation
#   All three describe the same thing (fields + types + which are required). This
#   one is the "lowest level" — it's the actual format the provider consumes.
#   LangChain converts TypedDict and Pydantic INTO json schema anyway; here you
#   just write it directly.
#
# WHY NO method="json_schema" THIS TIME
#   When you pass a Pydantic CLASS, ChatHuggingFace defaults to function_calling
#   and rejects it — so you had to force method="json_schema".
#   When you pass a raw dict schema like this, it already goes down the
#   json_schema path, so the explicit method= isn't needed. (Passing it anyway
#   does no harm.)
#
# JSON SCHEMA vs THE PYTHON WAYS — vocabulary map
#   Python / Pydantic              JSON Schema equivalent
#   ----------------------------   -------------------------------------------
#   list[str]                      {"type": "array", "items": {"type": "string"}}
#   Literal["pos","neg"]           {"type": "string", "enum": ["pos","neg"]}
#   Optional[str] (nullable)       {"type": ["string", "null"]}
#   required field                 name listed in the top-level "required" array
#   Field(description=...)         "description": "..." on that property
#   int                            {"type": "integer"}   (whole numbers)
#   float                          {"type": "number"}    (any number)
#
# HOW "required" WORKS HERE
#   required = ["key_themes", "summary", "sentiment"] -> these MUST be produced.
#   pros / cons / name are NOT in the list -> optional. Their "null" in the type
#   array is what lets the model legitimately return null when the review doesn't
#   supply them (this review has no email field and the name is present, so you'll
#   get name filled and, if absent, others as null).
#
# THE RETURN TYPE — it's a DICT
#   A raw JSON-schema schema behaves like the TypedDict path: invoke() gives back
#   a plain dict, NOT a validated object. So read result['name'],
#   result['sentiment'], etc. If you want validation, feed the dict into a
#   Pydantic model yourself: Review(**result).
#
# SAME PROVIDER CAVEAT AS BEFORE
#   Structured output is provider-sensitive on the HF stack. Cerebras 400s on it;
#   fireworks-ai (or together) honors the json_schema path. That's why the
#   provider is fireworks-ai here, not cerebras.
#
# WHEN YOU'D ACTUALLY REACH FOR RAW JSON SCHEMA
#   - The schema lives in an external .json file or comes from another system.
#   - You're not in Python-land / don't want a Pydantic dependency.
#   Otherwise Pydantic is usually nicer (validation + real objects). This style is
#   about interoperability and control over the exact wire format.
# ─────────────────────────────────────────────────────────────────────────────