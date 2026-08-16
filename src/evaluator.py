import json
from pathlib import Path

from sentence_transformers import SentenceTransformer, CrossEncoder

from document_loader import load_pdf
from fixed_chunker import fixed_size_chunk
from semantic_chunker import semantic_chunk_document
from retriever import create_index, search
from reranker import rerank_results


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def load_questions(path):
    with open(path, "r", encoding="utf-8") as file:
        questions = json.load(file)

    return [
        q for q in questions
        if q["relevant_pages"]
    ]


def calculate_metrics(results, relevant_pages):

    retrieved_pages = [
        result["page"]
        for result in results
    ]

    hit = any(
        page in relevant_pages
        for page in retrieved_pages
    )

    reciprocal_rank = 0

    for rank, page in enumerate(
        retrieved_pages,
        start=1
    ):
        if page in relevant_pages:
            reciprocal_rank = 1 / rank
            break

    return hit, reciprocal_rank, retrieved_pages


def evaluate_retrieval(
    questions,
    chunks,
    index,
    model,
    top_k=5
):

    all_results = []
    hits = []
    reciprocal_ranks = []

    for q in questions:

        retrieved = search(
            q["question"],
            index,
            chunks,
            model,
            top_k=top_k
        )

        hit, rr, pages = calculate_metrics(
            retrieved,
            q["relevant_pages"]
        )

        hits.append(int(hit))
        reciprocal_ranks.append(rr)

        all_results.append({
            "id": q["id"],
            "question": q["question"],
            "relevant_pages": q["relevant_pages"],
            "retrieved_pages": pages,
            "hit": hit,
            "reciprocal_rank": rr
        })

    return (
        all_results,
        sum(hits) / len(hits),
        sum(reciprocal_ranks) / len(reciprocal_ranks)
    )


def evaluate_reranking(
    questions,
    chunks,
    index,
    embedding_model,
    reranker,
    candidate_k=15,
    top_k=5
):

    all_results = []
    hits = []
    reciprocal_ranks = []

    for q in questions:

        # First retrieve a broader candidate set
        candidates = search(
            q["question"],
            index,
            chunks,
            embedding_model,
            top_k=candidate_k
        )

        # Then rerank candidates
        reranked = rerank_results(
            q["question"],
            candidates,
            reranker,
            top_k=top_k
        )

        hit, rr, pages = calculate_metrics(
            reranked,
            q["relevant_pages"]
        )

        hits.append(int(hit))
        reciprocal_ranks.append(rr)

        all_results.append({
            "id": q["id"],
            "question": q["question"],
            "relevant_pages": q["relevant_pages"],
            "retrieved_pages": pages,
            "hit": hit,
            "reciprocal_rank": rr
        })

    return (
        all_results,
        sum(hits) / len(hits),
        sum(reciprocal_ranks) / len(reciprocal_ranks)
    )


if __name__ == "__main__":

    pdf_path = Path("data/raw/apple_2025_10k.pdf")
    questions_path = Path("evaluation/test_questions.json")

    print("Loading evaluation questions...")
    questions = load_questions(questions_path)

    print(f"Questions with ground truth: {len(questions)}")

    print("\nLoading Apple 10-K...")
    pages = load_pdf(pdf_path)

    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # ---------------- FIXED ----------------

    print("\nCreating fixed chunks...")

    fixed_chunks = fixed_size_chunk(
        pages,
        chunk_size=1000,
        overlap=200
    )

    fixed_index = create_index(
        fixed_chunks,
        model
    )

    # ---------------- SEMANTIC ----------------

    print("\nCreating semantic chunks...")

    semantic_chunks = semantic_chunk_document(
        pages,
        model
    )

    semantic_index = create_index(
        semantic_chunks,
        model
    )

    # ---------------- EVALUATION ----------------

    print("\nEvaluating fixed-size retrieval...")

    fixed_results, fixed_hit, fixed_mrr = evaluate_retrieval(
        questions,
        fixed_chunks,
        fixed_index,
        model
    )

    print("Evaluating semantic retrieval...")

    semantic_results, semantic_hit, semantic_mrr = evaluate_retrieval(
        questions,
        semantic_chunks,
        semantic_index,
        model
    )

    print("Loading reranker...")

    reranker = CrossEncoder(RERANKER_MODEL)

    print("Evaluating semantic retrieval + reranking...")

    reranked_results, reranked_hit, reranked_mrr = evaluate_reranking(
        questions,
        semantic_chunks,
        semantic_index,
        model,
        reranker
    )

    # ---------------- SUMMARY ----------------

    print("\n" + "=" * 70)
    print("FINAL RETRIEVAL EVALUATION")
    print("=" * 70)

    print(f"\n{'METHOD':<30} {'HIT@5':<12} {'MRR':<12}")
    print("-" * 55)

    print(
        f"{'Fixed-size':<30}"
        f"{fixed_hit:<12.3f}"
        f"{fixed_mrr:<12.3f}"
    )

    print(
        f"{'Semantic':<30}"
        f"{semantic_hit:<12.3f}"
        f"{semantic_mrr:<12.3f}"
    )

    print(
        f"{'Semantic + Reranking':<30}"
        f"{reranked_hit:<12.3f}"
        f"{reranked_mrr:<12.3f}"
    )

    # ---------------- QUESTION DETAILS ----------------

    print("\n--- RERANKING DETAILS ---")

    for before, after in zip(
        semantic_results,
        reranked_results
    ):

        print("\n" + "-" * 70)

        print(
            f"{before['id']}: "
            f"{before['question']}"
        )

        print(
            f"Ground truth: "
            f"{before['relevant_pages']}"
        )

        print(
            f"Before reranking → "
            f"{before['retrieved_pages']} "
            f"| RR: {before['reciprocal_rank']:.3f}"
        )

        print(
            f"After reranking  → "
            f"{after['retrieved_pages']} "
            f"| RR: {after['reciprocal_rank']:.3f}"
        )