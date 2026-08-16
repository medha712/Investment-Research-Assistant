from pathlib import Path
from document_loader import load_pdf


pdf_path = Path("data/raw/apple_2025_10k.pdf")
pages = load_pdf(pdf_path)


while True:

    print("\nOPTIONS")
    print("1 - View a page")
    print("2 - Search the document")
    print("q - Quit")

    choice = input("\nChoose an option: ").strip()

    if choice.lower() == "q":
        break

    # VIEW PAGE
    if choice == "1":

        page_number = input("Enter page number: ")

        try:
            page_number = int(page_number)

            if 1 <= page_number <= len(pages):

                print("\n" + "=" * 80)
                print(f"PAGE {page_number}")
                print("=" * 80)

                print(pages[page_number - 1]["text"])

            else:
                print("Invalid page number.")

        except ValueError:
            print("Please enter a valid number.")

    # SEARCH DOCUMENT
    elif choice == "2":

        keyword = input("Enter search term: ").strip().lower()

        matches = []

        for page in pages:

            if keyword in page["text"].lower():
                matches.append(page["page"])

        if matches:
            print(f"\nFound '{keyword}' on pages:")
            print(matches)

        else:
            print("\nNo matches found.")

    else:
        print("Invalid option.")