# Retrievers — Smarter Ways to Fetch the Right Documents

## What a Retriever Is

A **retriever** is any object that takes a query string and hands back relevant
`Document`s. That's the whole contract:

```
retriever.invoke("some question")  ->  [Document, Document, ...]
```

Nothing more. It doesn't have to use embeddings, doesn't have to use a vector
store — it just has to answer "given this query, which documents matter?" That
uniform interface is the point: every retriever below plugs into a chain the same
way, so you can swap one for another without touching the rest of your code. This
is the **"R" in RAG** — the piece that fetches context for an LLM to answer from.

The story of this topic is the same shape as the text-splitting one: you start
with plain similarity search, then each new retriever fixes a specific weakness of
the last.

## Chapter 0 — The Odd One Out: Wikipedia Retriever

Before the vector-store family, one retriever proves the interface is general.
`WikipediaRetriever` fetches live from Wikipedia's own search — no embeddings, no
vector store, no API token of your own.

```python
retriever = WikipediaRetriever(top_k_results=2, lang="en")
docs = retriever.invoke("geopolitical history of india and pakistan")
```

It returns Wikipedia article chunks as `Document`s, exactly like every other
retriever. The lesson: "retriever" is an abstraction, not a synonym for "vector
search." An external source behind the same `.invoke()` interface is a valid
retriever.

## Chapter 1 — The Baseline: Vector Store Retriever

The workhorse. Take a vector store (Chroma, FAISS, etc.) and call `.as_retriever()`
on it — now the store speaks the standard retriever interface.

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
results = retriever.invoke("What is Chroma used for?")
```

This returns the same thing as calling `vectorstore.similarity_search(query, k=2)`
directly. So why bother? **Interface.** `similarity_search` is a raw method on one
specific store; `as_retriever()` gives you a standard, chainable object that drops
into a RAG pipeline and can be swapped for any other retriever. Everything below
is a smarter variant of this baseline.

**Its weakness:** plain similarity returns the *k nearest* documents — which can
be k near-duplicates, or padded with irrelevant text, or blind to how the question
was phrased. The next three retrievers each fix one of those.

## Chapter 2 — MMR: Fixing Redundancy

Plain similarity might return three documents that all say almost the same thing.
**MMR (Maximal Marginal Relevance)** fixes that by picking results that are
relevant *and also different from each other*.

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",                          # switch from plain similarity to MMR
    search_kwargs={"k": 3, "lambda_mult": 0.5}  # relevance <-> diversity dial
)
```

`lambda_mult` is the dial: 1.0 = pure relevance (may repeat), 0.0 = maximum
diversity, 0.5 balances both. Use MMR when your corpus has lots of overlapping
content and you want broad coverage instead of three copies of the same answer.

**Fixes:** redundant / near-duplicate results.

## Chapter 3 — MultiQuery: Fixing Bad Phrasing

A single query only searches from one angle. If the user phrases it differently
than the documents do, relevant results get missed. **MultiQueryRetriever** uses
an LLM to rewrite the question several ways, searches with each version, and merges
the unique hits.

```python
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=model      # needs an LLM to generate the rewordings
)
```

Ask "How to improve energy levels and maintain balance?" and the LLM might also
search "ways to boost energy", "maintaining physical balance", "staying healthy" —
catching documents a single phrasing would have missed.

**Fixes:** one-angle queries missing relevant docs. **Cost:** extra LLM calls
(this retriever needs a chat model, unlike the first three).

## Chapter 4 — ContextualCompression: Fixing Noisy Documents

The base retriever returns *whole* documents — and a document might be one relevant
sentence buried in paragraphs of unrelated text. **ContextualCompressionRetriever**
retrieves normally, then runs each result through a compressor (an LLM) that
extracts *only* the sentences relevant to the query.

```python
compressor = LLMChainExtractor.from_llm(model)    # the LLM that strips out irrelevant sentences
compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,   # fetches full (noisy) docs
    base_compressor=compressor       # trims each down to the relevant parts
)
```

Ask "What is photosynthesis?" against documents that mix photosynthesis facts with
sentences about the Grand Canyon and medieval castles, and you get back just the
photosynthesis lines — not the whole padded paragraph.

**Fixes:** relevant docs cluttered with irrelevant text. **Cost:** an LLM call per
document to compress it.

## The Whole Story in One Table

```
RETRIEVER               STRATEGY                          FIXES                     USES LLM?
----------------------  --------------------------------  ------------------------  ---------
Wikipedia               fetch live from Wikipedia         (external source)         no
Vector Store (baseline) k nearest by similarity           nothing — the baseline    no
MMR                     relevant BUT diverse              redundant results         no
MultiQuery              LLM rewrites query many ways      one-angle phrasing miss   yes
ContextualCompression   LLM trims docs to relevant parts  noisy padded documents    yes
```

## Environment Notes (from building this)

- **Embeddings/LLM run on Hugging Face**, not OpenAI (no credits). Only MultiQuery
  and ContextualCompression need the LLM (`ChatHuggingFace`); Wikipedia, Vector
  Store, and MMR are pure retrieval and touch no chat model.
- **Import relocations on langchain 1.x:** `langchain_community.*` and
  `langchain_core.*` stay as-is, but the classic retriever utilities moved —
  `MultiQueryRetriever`, `ContextualCompressionRetriever`, and `LLMChainExtractor`
  now import from **`langchain_classic.retrievers.*`** (install `langchain-classic`).
  `WikipediaRetriever` stays in `langchain_community.retrievers`.

## The Takeaway

**A retriever is a standard interface, and each type is a smarter layer on top of
plain similarity search.** MMR adds diversity, MultiQuery adds phrasing-robustness,
ContextualCompression adds result cleanup, and Wikipedia shows the interface even
covers external sources. Pick the plain vector-store retriever by default; reach
for the others when its specific weakness — redundancy, phrasing, or noise — is
actually hurting your results. Wire any of them into a chain with an LLM and you
have retrieval-augmented generation.
