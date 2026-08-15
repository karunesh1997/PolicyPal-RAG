from typing import List, Dict

from google import genai
from google.genai import types

from config.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL
)


SYSTEM_PROMPT = """
You are PolicyPal, an enterprise policy assistant.

Your job is to answer questions ONLY using the
provided policy context.

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer cannot be found in the context,
   clearly say that the information was not found
   in the available company policies.
4. Be concise and professional.
5. Explain the answer clearly.
6. Mention the relevant policy document and page
   when possible.
7. If multiple policies conflict, explicitly mention
   the conflict.
8. Never present unsupported information as fact.

The retrieved policy context is authoritative for
your answer.
"""


class Generator:

    def __init__(self):

        if not GEMINI_API_KEY:

            raise ValueError(
                "GEMINI_API_KEY is not configured. "
                "Add it to your .env file."
            )

        self.client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        self.model = GEMINI_MODEL

    def build_context(
        self,
        results: List[Dict]
    ) -> str:

        if not results:
            return ""

        context_parts = []

        for i, result in enumerate(
            results,
            start=1
        ):

            context_parts.append(
                f"""
SOURCE {i}

Document: {result['source']}
Page: {result['page']}

Content:
{result['text']}
"""
            )

        return "\n".join(context_parts)

    def answer(
        self,
        question: str,
        results: List[Dict]
    ) -> str:

        if not results:

            return (
                "I couldn't find enough relevant information "
                "in the available policies to answer this question."
            )

        context = self.build_context(results)

        prompt = f"""
{SYSTEM_PROMPT}

POLICY CONTEXT:
{context}

USER QUESTION:
{question}

Answer the question using ONLY the policy context.

If the context does not contain enough information,
say so clearly.

Keep the answer concise and easy to understand.
"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1
            )
        )

        if not response.text:

            return (
                "I wasn't able to generate an answer "
                "from the retrieved policy information."
            )

        return response.text.strip()

    def check_connection(self) -> bool:

        try:

            response = self.client.models.generate_content(
                model=self.model,
                contents="Respond with the word: OK"
            )

            return bool(response.text)

        except Exception:

            return False