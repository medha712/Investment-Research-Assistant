from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy


# --------------------------------------------------
# CREATE FAISS INDEX
# --------------------------------------------------

def create_index(chunks, embeddings):

    if not chunks:
        raise ValueError(
            "Cannot create FAISS index: "
            "no chunks were provided."
        )

    texts = []
    metadatas = []

    for chunk in chunks:

        text = chunk.get("text", "").strip()

        if text:
            texts.append(text)
            metadatas.append({"page": chunk["page"]})

    if not texts:
        raise ValueError(
            "Cannot create FAISS index: "
            "the chunks contain no usable text."
        )

    print(
        f"Creating embeddings for {len(texts)} chunks..."
    )

    # Cosine distance matches the normalized inner-product
    # similarity the original hand-rolled FAISS index used.
    vectorstore = FAISS.from_texts(
        texts,
        embeddings,
        metadatas=metadatas,
        distance_strategy=DistanceStrategy.COSINE
    )

    print(
        f"FAISS index created with "
        f"{vectorstore.index.ntotal} vectors."
    )

    return vectorstore


# --------------------------------------------------
# SEARCH FAISS INDEX
# --------------------------------------------------

def search(query, index, top_k=5):

    if not query or not query.strip():
        return []

    if index is None:
        raise ValueError(
            "FAISS index has not been created."
        )

    top_k = min(
        top_k,
        index.index.ntotal
    )

    if top_k <= 0:
        return []

    scored_documents = index.similarity_search_with_score(
        query,
        k=top_k
    )

    results = []

    for document, score in scored_documents:

        results.append({
            "text": document.page_content,
            "page": document.metadata["page"],
            "score": float(score)
        })

    return results
