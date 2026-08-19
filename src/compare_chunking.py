from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings

from document_loader import load_pdf
from fixed_chunker import fixed_size_chunk
from semantic_chunker import semantic_chunk_document
from retriever import create_index, search


MODEL_NAME = "all-MiniLM-L6-v2"


def print_results(title, results):

    print("\n")
    print("=" * 80)
    print(title)
    print("=" * 80)

    for number, result in enumerate(results, start=1):

        print(f"\nRESULT {number}")
        print(f"Similarity: {result['score']:.4f}")
        print(f"Page: {result['page']}")

        # Only print first 500 characters so terminal stays readable
        print(result["text"][:500])

        print("-" * 80)


if __name__ == "__main__":

    pdf_path = Path("data/raw/apple_2025_10k.pdf")

    print("Loading Apple 10-K...")
    pages = load_pdf(pdf_path)

    print("Loading embedding model...")
    model = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    # -------------------------
    # FIXED CHUNKING
    # -------------------------

    print("\nCreating fixed-size chunks...")

    fixed_chunks = fixed_size_chunk(
        pages,
        chunk_size=1000,
        overlap=200
    )

    print(f"Fixed chunks: {len(fixed_chunks)}")

    fixed_index = create_index(
        fixed_chunks,
        model
    )

    # -------------------------
    # SEMANTIC CHUNKING
    # -------------------------

    print("\nCreating semantic chunks...")

    semantic_chunks = semantic_chunk_document(
        pages,
        model
    )

    print(f"Semantic chunks: {len(semantic_chunks)}")

    semantic_index = create_index(
        semantic_chunks,
        model
    )

    # -------------------------
    # SAME QUESTION
    # -------------------------

    query = "What are the major risks facing Apple?"

    print("\nQUESTION:")
    print(query)

    fixed_results = search(
        query,
        fixed_index,
        top_k=5
    )

    semantic_results = search(
        query,
        semantic_index,
        top_k=5
    )

    # -------------------------
    # RESULTS
    # -------------------------

    print_results(
        "FIXED-SIZE CHUNKING RESULTS",
        fixed_results
    )

    print_results(
        "SEMANTIC CHUNKING RESULTS",
        semantic_results
    )
