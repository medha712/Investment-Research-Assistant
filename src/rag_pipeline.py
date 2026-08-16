from pathlib import Path

from sentence_transformers import SentenceTransformer, CrossEncoder

from document_loader import load_pdf
from semantic_chunker import semantic_chunk_document
from retriever import create_index, search
from reranker import rerank_results
from generator import generate_answer


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

# Default document used when running this file directly
PDF_PATH = Path("data/raw/apple_2025_10k.pdf")


# --------------------------------------------------
# RAG PIPELINE
# --------------------------------------------------

class InvestmentResearchRAG:

    def __init__(self, pdf_path=PDF_PATH):

        self.pdf_path = Path(pdf_path)

        # -------------------------------
        # 1. LOAD PDF
        # -------------------------------

        print(f"Loading document: {self.pdf_path}")

        self.pages = load_pdf(self.pdf_path)

        if not self.pages:
            raise ValueError(
                "No pages could be extracted from the PDF."
            )

        # Check whether extracted pages actually contain text
        text_pages = [
            page
            for page in self.pages
            if page.get("text", "").strip()
        ]

        if not text_pages:
            raise ValueError(
                "The PDF does not contain extractable text. "
                "It may be scanned or image-based."
            )

        print(
            f"Loaded {len(self.pages)} pages "
            f"({len(text_pages)} with extractable text)."
        )

        # -------------------------------
        # 2. EMBEDDING MODEL
        # -------------------------------

        print("Loading embedding model...")

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        # -------------------------------
        # 3. SEMANTIC CHUNKING
        # -------------------------------

        print("Creating semantic chunks...")

        self.chunks = semantic_chunk_document(
            self.pages,
            self.embedding_model
        )

        if not self.chunks:
            raise ValueError(
                "No text chunks were created from the document. "
                "The PDF may contain too little extractable text, "
                "or the semantic chunker could not process it."
            )

        print(
            f"Created {len(self.chunks)} semantic chunks."
        )

        # -------------------------------
        # 4. FAISS INDEX
        # -------------------------------

        print("Creating FAISS index...")

        self.index = create_index(
            self.chunks,
            self.embedding_model
        )

        # -------------------------------
        # 5. RERANKER
        # -------------------------------

        print("Loading reranker...")

        self.reranker = CrossEncoder(
            RERANKER_MODEL
        )

        print("\nInvestment Research RAG is ready!")


    # --------------------------------------------------
    # ASK QUESTION
    # --------------------------------------------------

    def ask(self, question):

        if not question or not question.strip():
            raise ValueError(
                "Please enter a question."
            )

        # -------------------------------
        # STEP 1 — RETRIEVE
        # -------------------------------

        candidates = search(
            question,
            self.index,
            self.chunks,
            self.embedding_model,
            top_k=15
        )

        if not candidates:
            return {
                "question": question,
                "answer": (
                    "I couldn't find relevant information "
                    "in the uploaded document."
                ),
                "sources": []
            }

        # -------------------------------
        # STEP 2 — RERANK
        # -------------------------------

        reranked_results = rerank_results(
            question,
            candidates,
            self.reranker,
            top_k=5
        )

        if not reranked_results:
            return {
                "question": question,
                "answer": (
                    "I couldn't find sufficiently relevant "
                    "evidence in the uploaded document."
                ),
                "sources": []
            }

        # -------------------------------
        # STEP 3 — GENERATE ANSWER
        # -------------------------------

        answer = generate_answer(
            question,
            reranked_results
        )

        return {
            "question": question,
            "answer": answer,
            "sources": reranked_results
        }


# --------------------------------------------------
# TERMINAL TEST
# --------------------------------------------------

if __name__ == "__main__":

    rag = InvestmentResearchRAG()

    while True:

        question = input(
            "\nAsk a question about the document "
            "(or type 'quit'): "
        )

        if question.lower().strip() in [
            "quit",
            "exit",
            "q"
        ]:
            print("Goodbye!")
            break

        try:

            print("\nSearching document...")

            result = rag.ask(question)

            print("\n" + "=" * 70)
            print("ANSWER")
            print("=" * 70)

            print(result["answer"])

            print("\n" + "=" * 70)
            print("SOURCES")
            print("=" * 70)

            seen_pages = set()

            for source in result["sources"]:

                page = source["page"]

                if page not in seen_pages:

                    print(
                        f"Document — Page {page}"
                    )

                    seen_pages.add(page)

        except Exception as error:

            print(
                f"\nError: {error}"
            )