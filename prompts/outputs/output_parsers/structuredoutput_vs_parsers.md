# Structured Output vs Output Parsers — When to Use What

## The Core Problem

An LLM, left alone, returns free-form text. But real programs need *data* — a
dict with known keys, a field that's guaranteed to be an integer, a value that's
one of a fixed set. Parsing that out of a paragraph by hand is fragile. LangChain
gives two different tools that both solve this same problem, and they're easy to
confuse because the end result looks identical. The difference is entirely about
**who does the work of forcing the structure.**

## Two Solutions, Same Goal

```
with_structured_output   → the PROVIDER'S API enforces the shape
                           (tool-calling or native json_schema mode)

output parsers           → YOU enforce the shape, by injecting
                           "return JSON like this" into the prompt
                           and parsing the reply yourself
```

`with_structured_output` hands the job to the model's hosting API. When that API
supports it, this is the cleaner path — you declare a schema, the provider
guarantees the model fills it. Output parsers ask the API for nothing special;
they are pure prompt-in, text-out, parse. Because they rely only on words and
string-handling, they work against **any** model, anywhere.

**Decision: `with_structured_output` is the cleaner option when the provider
cooperates; output parsers are the universal fallback that always works.**

## Topic 1 — `with_structured_output` (three schema styles)

Within this topic, the only thing that changes between the three approaches is
*how you describe the shape you want*. They are three dialects of the same
instruction.

- **TypedDict** — the lightest. Field names and types as a hint, nothing more. No
  validation, returns a plain dict. Reach for it when you trust the model and
  want minimal code. It also happens to be the one that works on the
  `function_calling` path when a Pydantic class is rejected.
- **Pydantic** — the default choice, and what most production code uses. Real
  validation, constraints like `gt` / `lt`, type coercion, and it returns a
  proper validated object. Use it whenever you want guarantees on the output.
- **JSON Schema (raw dict)** — the lowest-level form, the one the API actually
  consumes underneath. Use it when the schema lives in a `.json` file or arrives
  from a non-Python source. No validation, returns a dict.

Pydantic is the everyday pick; the other two exist for specific situations
(minimalism, or an externally-defined schema).

## Topic 2 — Output Parsers (four, in increasing power)

- **StrOutputParser** — not structured at all. It simply extracts `.content` as a
  plain string. Its purpose is to save you from writing `.content` by hand inside
  a chain. This is the parser you'll use *most often* overall — just not for
  structure.
- **JsonOutputParser** — returns valid JSON (a dict or list), but you do **not**
  control the shape; the model decides the keys. Use it when you want "some JSON"
  and don't care about exact fields.
- **StructuredOutputParser** — fixed keys that you declare, but every value comes
  back as a string and nothing is validated. Now legacy (moved to
  `langchain_classic`). Skippable in practice — Pydantic does everything it does
  and more.
- **PydanticOutputParser** — fixed shape **plus** types **plus** validation,
  returning a validated object. The best parser, and the one to reach for when
  you need real structure but the provider won't do `with_structured_output`.

## The Nuance That Matters — "Can" Means the Provider, Not the Model

The common way to summarize the two topics is: **use Topic 1 for LLMs that *can*
give structured output, and Topic 2 for LLMs that *can't*.** That rule of thumb
is correct — but the word "can" hides the single most important detail, and
getting it wrong will send you debugging in the wrong place.

The capability is **not a property of the model.** It is a property of the
**provider / API endpoint the model is served through.** The same model can land
on *either* side of the can / can't line depending purely on where it's hosted.
A model doesn't "know how" to do structured output in some innate way — the
endpoint in front of it either exposes the tool-calling / json_schema machinery
or it doesn't.

This is exactly what happened while building this repo, and it's worth keeping as
the concrete example:

```
Model: openai/gpt-oss-120b   (a fully capable model, unchanged)

  via Cerebras   → the API rejects the structured-output request → 400 error
                 → with_structured_output is NOT usable here
                 → forced to drop to Topic 2 (PydanticOutputParser)

  via Fireworks  → the API exposes json_schema → structured output works
                 → with_structured_output is usable here
```

Same model, same schema, same code — the *only* variable was the provider, and it
flipped the answer to "can this do Topic 1?" from no to yes. If you read the rule
as "gpt-oss-120b can't do structured output," you'd draw the wrong conclusion;
the truth is "gpt-oss-120b **on Cerebras** can't, because that provider doesn't
expose it."

**Decision: read the tutor's "can / can't" as "does *this provider's API*
support it," not "is *this model* capable." A theoretically-capable model still
lands you in Topic 2 whenever the provider you're routing through doesn't expose
the feature.**

This is also *why* Topic 2 exists at all and why it's the safe default under
pressure. Prompt-and-parse works regardless of what the API supports, because it
never touches the provider's native structured-output path — it only sends text
and reads text back. That independence is the whole point: it's the escape hatch
for exactly the situation above.

## Decision Guide — What to Use When

- **Just need the text out of a chain?** → `StrOutputParser` (most common overall).
- **Need structured data, and the provider supports it well** (OpenAI, Fireworks,
  etc.)? → `with_structured_output(PydanticModel)` — cleanest.
- **Need structured data, but the provider chokes on it** (the Cerebras case)? →
  `PydanticOutputParser` — same schema, same validation, works anywhere.
- **Need loose JSON and don't care about exact keys?** → `JsonOutputParser`.
- **`StructuredOutputParser`, `TypedDict`, raw `JSON Schema`** → know they exist,
  but you'll rarely reach for them; Pydantic covers their ground better in almost
  every case.

## The Through-Line

**Pydantic is the spine of both topics.** Whether you go the
`with_structured_output` route or the parser route, a Pydantic schema is the
answer in both — one lets the *provider* enforce it, the other enforces it via
*prompt + parse*. Learn Pydantic well and you've learned the core of both
topics; everything else is a variation for a narrower situation.

**Rule of thumb: try `with_structured_output` first** (less code, the provider
does the work). **The moment you hit a provider error, fall back to
`PydanticOutputParser`** — it is the portable version of the exact same thing.
That single fallback is the entire lesson of the errors fought through while
building this folder.