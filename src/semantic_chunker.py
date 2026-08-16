from pathlib import Path
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from document_loader import load_pdf


MODEL_NAME = "all-MiniLM-L6-v2"


def split_into_sentences(text):
    """Simple sentence splitter."""
    sentences = re.split(r'(?<=[.!?])\s+', text)

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def semantic_chunk_page(
    text,
    model,
    similarity_threshold=0.55,
    max_chunk_chars=2000
):
    sentences = split_into_sentences(text)

    if not sentences:
        return []

    if len(sentences) == 1:
        return [sentences[0]]

    # Convert every sentence into an embedding.
    embeddings = model.encode(
        sentences,
        normalize_embeddings=True
    )

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):

        previous_embedding = embeddings[i - 1].reshape(1, -1)
        current_embedding = embeddings[i].reshape(1, -1)

        similarity = cosine_similarity(
            previous_embedding,
            current_embedding
        )[0][0]

        current_text = " ".join(current_chunk)

        # Start a new chunk when the meaning changes significantly
        # or when the chunk becomes too large.
        if (
            similarity < similarity_threshold
            or len(current_text) + len(sentences[i]) > max_chunk_chars
        ):
            chunks.append(current_text)
            current_chunk = [sentences[i]]

        else:
            current_chunk.append(sentences[i])

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def semantic_chunk_document(pages, model):

    chunks = []

    for page in pages:

        page_chunks = semantic_chunk_page(
            page["text"],
            model
        )

        for chunk_text in page_chunks:

            chunks.append({
                "text": chunk_text,
                "page": page["page"],
                "chunk_type": "semantic"
            })

    return chunks


if __name__ == "__main__":

    pdf_path = Path("data/raw/apple_2025_10k.pdf")

    print("Loading Apple 10-K...")
    pages = load_pdf(pdf_path)

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Creating semantic chunks...")
    chunks = semantic_chunk_document(pages, model)

    print(f"\nTotal pages: {len(pages)}")
    print(f"Total semantic chunks: {len(chunks)}")

    print("\n--- SEMANTIC CHUNK 1 ---\n")
    print(chunks[0]["text"])
    print("\nPage:", chunks[0]["page"])

    print("\n--- SEMANTIC CHUNK 2 ---\n")
    print(chunks[1]["text"])
    print("\nPage:", chunks[1]["page"])