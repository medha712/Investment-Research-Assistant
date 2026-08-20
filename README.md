# Investment Research Assistant

**An Evaluation-Driven RAG System for Financial Documents**

This project is a Streamlit application for asking natural-language questions about financial PDFs. It retrieves relevant evidence, reranks it, and asks Gemini to produce a grounded answer with page-level citations. The primary development and evaluation document is Apple's 2025 Form 10-K.

![Investment Research Assistant workspace](docs/screenshots/application-research-workspace.png)

## Features

- PDF upload and text extraction with PyMuPDF
- Semantic chunking with sentence embeddings
- FAISS dense-vector retrieval
- CrossEncoder reranking
- Gemini answers constrained to retrieved evidence
- Page citations, evidence previews, and chat history
- Fixed-size versus semantic retrieval evaluation using Hit@5 and MRR

## Architecture

```mermaid
flowchart TD
    U["User"] --> UI["Streamlit frontend"]
    UI --> PDF["Financial PDF"]
    PDF --> L["LangChain PyMuPDFLoader"]
    L --> SC["Semantic chunking<br/>0.55 cosine threshold"]
    SC --> E["LangChain HuggingFaceEmbeddings<br/>all-MiniLM-L6-v2"]
    E --> F["LangChain FAISS vectorstore<br/>cosine distance"]
    F --> C["Top 15 candidate chunks"]
    C --> R["CrossEncoder reranker<br/>ms-marco-MiniLM-L6-v2"]
    R --> T["Top 5 evidence chunks"]
    T --> G["Gemini 3.6-flash<br/>grounded generation"]
    G --> A["Answer with Page citations"]
    A --> UI

    FX["Fixed-size baseline"] --> EV["Evaluation<br/>Hit@5 and MRR"]
    SM["Semantic chunking<br/>original algorithm"] --> EV
    SR["Semantic + reranking"] --> EV

## Technology stack

Python, Streamlit, PyMuPDF, SentenceTransformers, scikit-learn, NumPy, FAISS, Google GenAI SDK, Gemini, and python-dotenv.

## Document and retrieval methodology

The development PDF is an approximately 80-page Apple 2025 Form 10-K. Ten evaluation questions are stored in `evaluation/test_questions.json`; eight currently have manually validated relevant pages and are included by `src/evaluator.py`.

The final application uses semantic chunks. Sentence-level embeddings are compared between adjacent sentences; a chunk boundary is created below a similarity threshold of `0.55` or at a maximum of 2,000 characters. FAISS retrieves 15 candidates and a CrossEncoder reranks them to the best five evidence chunks.

The experimental baseline uses 1,000-character chunks with 200-character overlap.

## Retrieval results

| Method | Hit@5 | MRR |
|---|---:|---:|
| Fixed-size | 1.000 | 0.713 |
| Semantic | 1.000 | 0.854 |
| Semantic + Reranking | 1.000 | 0.875 |

Hit@5 measures whether at least one relevant page appears in the first five results. MRR rewards placing the first relevant page nearer the top. Because Hit@5 was 1.000 for all three approaches, MRR was the more useful discriminator in this eight-question evaluation. These results are specific to this small dataset and do not establish universal superiority.

## Setup

The inspected development environment uses Python 3.14.7. Compatibility with other Python versions should be verified in the intended submission or deployment environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```dotenv
GEMINI_API_KEY=your_api_key_here
```

Never commit `.env`. The current `.gitignore` excludes it.

## Run the application

```powershell
streamlit run app.py
```

Upload a text-based financial PDF, select **Process document**, and wait for indexing to complete.

Example questions:

- What are the major risks facing Apple?
- How did Apple's total net sales change in 2025?
- What factors affect Apple's gross margins?
- What does Apple say about competition in its markets?

## Run the evaluation

From the repository root:

```powershell
python src/evaluator.py
```

The script evaluates only questions with populated `relevant_pages` and prints per-method summaries and per-question reranking details.

## Repository structure

```text
investment-research-assistant/
├── app.py                         # Streamlit frontend and integration
├── data/raw/                      # Development PDF
├── data/uploads/                  # Runtime uploads; ignored by Git
├── evaluation/test_questions.json
├── src/
│   ├── document_loader.py
│   ├── fixed_chunker.py
│   ├── semantic_chunker.py
│   ├── retriever.py
│   ├── reranker.py
│   ├── generator.py
│   ├── rag_pipeline.py
│   ├── evaluator.py
│   ├── build_ground_truth.py
│   ├── compare_chunking.py
│   └── inspect_pages.py
├── docs/screenshots/               # Application screenshots
└── requirements.txt
```

## Limitations

- Retrieval evaluation uses one filing and eight validated questions.
- The generator prompt is currently specialized to Apple's 2025 Form 10-K.
- The index is rebuilt in memory whenever a document is processed.
- Image-only PDFs do not have an OCR path.
- Generation depends on an external Gemini API and the configured model's availability.
- Financial tables are handled as extracted text rather than through a dedicated table parser.

GitHub repository: [medha712/Investment-Research-Assistant](https://github.com/medha712/Investment-Research-Assistant)
