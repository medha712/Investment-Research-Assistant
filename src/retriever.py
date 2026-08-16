import faiss
import numpy as np


# --------------------------------------------------
# CREATE FAISS INDEX
# --------------------------------------------------

def create_index(chunks, model):

    if not chunks:
        raise ValueError(
            "Cannot create FAISS index: "
            "no chunks were provided."
        )

    texts = []

    for chunk in chunks:

        text = chunk.get("text", "").strip()

        if text:
            texts.append(text)

    if not texts:
        raise ValueError(
            "Cannot create FAISS index: "
            "the chunks contain no usable text."
        )

    print(
        f"Creating embeddings for {len(texts)} chunks..."
    )

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    # Make sure embeddings have the expected shape
    if embeddings.ndim != 2:
        raise ValueError(
            "Embedding generation failed. "
            f"Expected a 2D matrix but received "
            f"shape {embeddings.shape}."
        )

    if embeddings.shape[0] == 0:
        raise ValueError(
            "Embedding generation returned zero vectors."
        )

    # Convert to float32 for FAISS
    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    # Normalize embeddings so inner product
    # behaves like cosine similarity
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    print(
        f"FAISS index created with "
        f"{index.ntotal} vectors."
    )

    return index


# --------------------------------------------------
# SEARCH FAISS INDEX
# --------------------------------------------------

def search(
    query,
    index,
    chunks,
    model,
    top_k=5
):

    if not query or not query.strip():
        return []

    if index is None:
        raise ValueError(
            "FAISS index has not been created."
        )

    if not chunks:
        return []

    # We cannot retrieve more results
    # than actually exist in the index
    top_k = min(
        top_k,
        index.ntotal
    )

    if top_k <= 0:
        return []

    # Embed user query
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        show_progress_bar=False
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    faiss.normalize_L2(
        query_embedding
    )

    # Search FAISS
    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        # FAISS can return -1 when no result exists
        if idx == -1:
            continue

        if idx >= len(chunks):
            continue

        chunk = chunks[idx]

        results.append({
            "text": chunk["text"],
            "page": chunk["page"],
            "score": float(score)
        })

    return results