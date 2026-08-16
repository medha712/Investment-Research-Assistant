import os
from google import genai
from dotenv import load_dotenv


# Load variables from .env
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY was not found in .env")

client = genai.Client(api_key=API_KEY)


def generate_answer(query, retrieved_results):
    """
    Generate an answer using only the evidence retrieved
    from the Apple 2025 Form 10-K.
    """

    context_parts = []

    for result in retrieved_results:

        context_parts.append(
            f"""
SOURCE PAGE: {result['page']}

{result['text']}
"""
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are an investment research assistant.

Answer the user's question using ONLY the information contained
in the provided context from Apple's 2025 Form 10-K.

Rules:
1. Do not use outside knowledge.
2. Do not invent financial figures or facts.
3. If the context does not contain enough information to answer
   the question, clearly say that the available document evidence
   is insufficient.
4. Give a concise but useful investment-research answer.
5. When making factual claims, cite the relevant source page
   using the format [Page X].

USER QUESTION:
{query}

CONTEXT:
{context}

ANSWER:
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text