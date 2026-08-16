import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from document_loader import load_pdf
from semantic_chunker import semantic_chunk_document
from retriever import create_index, search


MODEL_NAME = "all-MiniLM-L6-v2"

PDF_PATH = Path("data/raw/apple_2025_10k.pdf")
QUESTIONS_PATH = Path("evaluation/test_questions.json")


def load_questions():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":

    # Load document
    print("Loading Apple 10-K...")
    pages = load_pdf(PDF_PATH)

    # Load questions
    print("Loading evaluation questions...")
    questions = load_questions()

    print(f"Questions loaded: {len(questions)}")

    # Load embedding model
    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    # Create semantic chunks
    print("Creating semantic chunks...")
    chunks = semantic_chunk_document(
        pages,
        model
    )

    print(f"Semantic chunks: {len(chunks)}")

    # Create FAISS index
    print("\nCreating search index...")
    index = create_index(
        chunks,
        model
    )

    print("\n" + "=" * 80)
    print("GROUND TRUTH CANDIDATE PAGES")
    print("=" * 80)

    # Go through every evaluation question
    for question_data in questions:

        question = question_data["question"]

        results = search(
            question,
            index,
            chunks,
            model,
            top_k=10
        )

        # Get unique candidate pages
        candidate_pages = []

        for result in results:
            page = result["page"]

            if page not in candidate_pages:
                candidate_pages.append(page)

        candidate_pages = candidate_pages[:5]

        # Print question
        print("\n" + "-" * 80)

        print(
            f"{question_data['id']}: "
            f"{question}"
        )

        print(f"Suggested pages: {candidate_pages}")

        # Print preview from each candidate page
        for page_number in candidate_pages:

            page_text = pages[page_number - 1]["text"]

            preview = page_text[:350].replace("\n", " ")

            print(f"\nPAGE {page_number}:")
            print(preview)