from pathlib import Path
import fitz


def load_pdf(pdf_source):
    """
    Load a PDF and extract text page by page.

    Returns:
        [
            {
                "page": 1,
                "text": "..."
            },
            ...
        ]
    """

    if isinstance(pdf_source, (bytes, bytearray, memoryview)):
        pdf_bytes = bytes(pdf_source)

        if not pdf_bytes:
            raise ValueError("The uploaded PDF is empty.")

        print("Opening PDF from uploaded bytes")
        document = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )
    else:
        pdf_path = Path(pdf_source)

        if not pdf_path.is_file():
            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        if pdf_path.stat().st_size == 0:
            raise ValueError(f"PDF is empty: {pdf_path}")

        print(f"Opening PDF: {pdf_path}")
        document = fitz.open(str(pdf_path))

    pages = []

    print(f"PDF contains {len(document)} pages.")

    try:
        for page_number in range(len(document)):

            page = document.load_page(page_number)

            # Primary extraction
            text = page.get_text("text")

            # Clean whitespace
            if text:
                text = text.strip()
            else:
                text = ""

            pages.append({
                "page": page_number + 1,
                "text": text
            })
    finally:
        document.close()

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
