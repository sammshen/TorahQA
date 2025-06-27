# Procedure for generating QA given context

Generate 25 questions at once with a prompt to GPT 4o (128k token context window):

```text
can you generate 25 extremely detailed (as fine grained as possible) question answer pairs in the JSON format (but textual output please) of: 

[
{"question": "", "answer": ""},
...
]

good questions:
1. which verse talks about... (answer will just be a verse prefixed by the book name e.g. "Gensis 1:1")
2. esoteric details about sacrifices, ordination, laws, codes, atonement, etc.

good answers:
1. should not include citations unless the question explicitly asks
2. should be a short answer and not give any details that are not asked for (this implies the question should ask for a short answer as well)

Here is the context:

[Chunk of the document that fits into GPT 4o's context window]
```