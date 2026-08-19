from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader


def load_pdf(pdf_path):
    """
    Load a PDF and extract text page by page using LangChain's
    PyMuPDFLoader (still backed by PyMuPDF/fitz under the hood).

    Returns:
        [
            {
                "page": 1,
                "text": "..."
            },
            ...
        ]
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    if pdf_path.stat().st_size == 0:
        raise ValueError(f"PDF is empty: {pdf_path}")

    print(f"Opening PDF: {pdf_path}")

    loader = PyMuPDFLoader(str(pdf_path))
    documents = loader.load()

    print(f"PDF contains {len(documents)} pages.")

    pages = []

    for document in documents:

        text = document.page_content.strip() if document.page_content else ""

        # PyMuPDFLoader pages are 0-indexed; keep 1-indexed pages
        # so citations match the physical page numbers shown to users.
        page_number = document.metadata.get("page", len(pages)) + 1

        pages.append({
            "page": page_number,
            "text": text
        })

    pages_with_text = sum(
        1
        for page in pages
        if page["text"].strip()
    )

    print(
        f"Text extracted from "
        f"{pages_with_text}/{len(pages)} pages."
    )

    return pages
