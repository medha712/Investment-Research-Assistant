from pathlib import Path
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings

from document_loader import load_pdf
from semantic_chunker import semantic_chunk_document
from retriever import create_index, search


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def rerank_results(query, results, reranker, top_k=5):
    """
    Rerank retrieved chunks using a CrossEncoder.
    """

    pairs = [
        [query, result["text"]]
        for result in results
    ]

    scores = reranker.predict(pairs)

    reranked = []

    for result, score in zip(results, scores):
        new_result = result.copy()
        new_result["rerank_score"] = float(score)
        reranked.append(new_result)

    reranked.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked[:top_k]


if __name__ == "__main__":

    pdf_path = Path("data/raw/apple_2025_10k.pdf")

    print("Loading Apple 10-K...")
    pages = load_pdf(pdf_path)

    print("Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Creating semantic chunks...")
    chunks = semantic_chunk_document(
        pages,
        embedding_model
    )

    print(f"Total chunks: {len(chunks)}")

    print("Creating FAISS index...")
    index = create_index(
        chunks,
        embedding_model
    )

    query = "What are the major risks facing Apple?"

    print("\nQUESTION:")
    print(query)

    # Retrieve more candidates than we ultimately need
    initial_results = search(
        query,
        index,
        top_k=15
    )

    print("\nLoading reranker...")
    reranker = CrossEncoder(RERANKER_MODEL)

    reranked_results = rerank_results(
        query,
        initial_results,
        reranker,
        top_k=5
    )

    print("\n" + "=" * 70)
    print("TOP 5 BEFORE RERANKING")
    print("=" * 70)

    for number, result in enumerate(initial_results[:5], start=1):

        print(f"\nRESULT {number}")
        print(f"Page: {result['page']}")
        print(f"Similarity: {result['score']:.4f}")
        print(result["text"][:500])

    print("\n" + "=" * 70)
    print("TOP 5 AFTER RERANKING")
    print("=" * 70)

    for number, result in enumerate(reranked_results, start=1):

        print(f"\nRESULT {number}")
        print(f"Page: {result['page']}")
        print(f"Reranker score: {result['rerank_score']:.4f}")
        print(result["text"][:500])
