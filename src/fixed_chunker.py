from document_loader import load_pdf
from pathlib import Path


def fixed_size_chunk(pages, chunk_size=1000, overlap=200):

    chunks = []

    for page in pages:

        text = page["text"]
        page_number = page["page"]

        start = 0

        while start < len(text):

            end = start + chunk_size
            chunk_text = text[start:end]

            chunks.append({
                "text": chunk_text,
                "page": page_number,
                "chunk_type": "fixed"
            })

            start += chunk_size - overlap

    return chunks


if __name__ == "__main__":

    pdf_path = Path("data/raw/apple_2025_10k.pdf")

    pages = load_pdf(pdf_path)

    chunks = fixed_size_chunk(
        pages,
        chunk_size=1000,
        overlap=200
    )

    print(f"Total pages: {len(pages)}")
    print(f"Total fixed-size chunks: {len(chunks)}")

    print("\n--- CHUNK 1 ---\n")
    print(chunks[0]["text"])
    print("\nPage:", chunks[0]["page"])

    print("\n--- CHUNK 2 ---\n")
    print(chunks[1]["text"])
    print("\nPage:", chunks[1]["page"])