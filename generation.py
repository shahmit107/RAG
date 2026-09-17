
def build_prompt(query_text, retrieved_chunks):
    # Step 1: join the retrieved chunks into one context block
    formated_chunks = []
    for index, chunk in enumerate(retrieved_chunks, start = 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "unknown")
        page = metadata.get("page", "N/A")
        text = f"[{index}] source: {source}, page: {page}, document: {chunk.get('document', '')}"
        formated_chunks.append(text)

    context = "\n\n".join(formated_chunks)
    #method to stitch them together, separated by double newlines so the LLM can easily distinguish between different documents.

    # Step 2: build the final prompt string using an f-string, following the pattern shown above
    prompt = f"""You are a precise research assistant. Answer the question using ONLY the information in the context below — no outside knowledge, no assumptions.
    
    Context is a list of chunks. Each chunk is preceded by a citation ID and its metadata in the form:
[1] source: <source>, page: <page>, document: <document>

    Context:
    {context}

    Question: {query_text}

    Rules:
    1. Base your answer strictly on the context above.
    2. If the context does not contain enough information to answer, respond with exactly: "I don't know based on the provided context." and nothing else.
    3. Only cite chunks you actually used to form the answer — never list a chunk you didn't rely on, and never invent a source or page number not present in the context.
    4. Where possible, quote the exact supporting phrase (short, under ~15 words) from the chunk so the citation can be matched back to the source text.
    5. Return your response as valid JSON only, no markdown fences, in this exact shape:

    {{
      "answer": "<your answer in plain text>",
      "citations": [
        {{"source": "<source>", "page": "<page>"}}
      ]
    }}

    If you don't know the answer, return "citations": [] and the fallback message as the "answer" value.
    """
    return prompt