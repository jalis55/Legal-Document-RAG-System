# src/generation/drafter.py
"""
Grounded Legal Draft Generator with Feedback Learning
"""

import os
from typing import List, Dict

from groq import Groq
from dotenv import load_dotenv
from src.retrieval.retriever import LegalRetriever
from src.feedback.feedback_handler import feedback_handler

load_dotenv()

class LegalDrafter:
    def __init__(self, model: str = None):
        if model is None:
            model = os.getenv("MODEL_NAME", "qwen/qwen3-32b")
        
        self.retriever = LegalRetriever()
        self.model = model
        
        # Validate API key early
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables. Please add to .env file")
        
        self.client = Groq(api_key=api_key)

    def generate_draft(self, query: str, top_k: int = 6, draft_type: str = "general_analysis") -> Dict:
        """Generate draft with optional learning from past feedback"""
        
        # 1. Retrieve relevant chunks
        retrieved_chunks = self.retriever.search(query, top_k=top_k)
        if not retrieved_chunks:
            return {"error": "No relevant content found in vector database. Please ensure documents are ingested."}

        context = self._build_context(retrieved_chunks)

        # 2. Get learning examples from feedback
        feedback_examples = feedback_handler.get_recent_feedback(limit=3)

        # 3. Build prompts
        system_prompt = self._get_system_prompt(draft_type, feedback_examples)
        user_prompt = self._get_user_prompt(query, context, draft_type)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.25,
                max_tokens=2000,
            )

            draft_text = response.choices[0].message.content

            return {
                "query": query,
                "draft_type": draft_type,
                "draft": draft_text,
                "sources": [
                    {
                        "filename": chunk.get("filename"),
                        "page_number": chunk.get("page_number"),
                        "relevance_score": round(chunk.get("relevance_score", 0), 3)
                    }
                    for chunk in retrieved_chunks
                ],
                "used_feedback": len(feedback_examples) > 0,
                "model_used": self.model
            }

        except Exception as e:
            return {"error": f"Groq API Error: {str(e)}"}

    def _build_context(self, chunks: List[Dict]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[Source {i}] File: {chunk.get('filename')}, Page {chunk.get('page_number')}\n"
                f"{chunk['text']}\n"
            )
        return "\n---\n".join(parts)

    def _get_system_prompt(self, draft_type: str, feedback_examples: List[Dict]) -> str:
        base = """You are an expert legal assistant at Pearson Specter Litt.
Generate professional, concise, and well-structured legal drafts.
Strictly base your response on the provided evidence only.
Do not hallucinate information or add external knowledge.
If the evidence doesn't fully answer the query, state what is missing."""

        # Add learning from feedback (Few-shot learning)
        if feedback_examples:
            base += "\n\n### Learning from Previous Operator Edits\n"
            base += "Below are examples of how an operator edited previous drafts. Learn from these improvements.\n"
            
            for i, fb in enumerate(feedback_examples, 1):
                base += f"\n--- Example {i} ---\n"
                base += f"Query: {fb.get('query')}\n"
                
                # Show key differences instead of full drafts
                original_preview = fb.get('original_draft', '')[:200]
                edited_preview = fb.get('edited_draft', '')[:200]
                base += f"Original: {original_preview}...\n"
                base += f"Improved: {edited_preview}...\n"
                
                if fb.get('user_notes'):
                    base += f"Operator Notes: {fb.get('user_notes')}\n"
            base += "\n--- End of Examples ---\n"

        # Draft type specific instructions
        if draft_type == "first_pass_internal_memo":
            base += "\n\nFormat as internal memo with: TO:, FROM:, DATE:, SUBJECT:, then Summary, Analysis, and Recommendations sections."
        elif draft_type == "document_checklist":
            base += "\n\nOutput as a markdown checklist with [ ] for incomplete items and [x] for completed where applicable."
        elif draft_type == "case_fact_summary":
            base += "\n\nOrganize as: Parties Involved, Key Dates, Material Facts, Procedural History, Current Status."
        elif draft_type == "title_review_summary":
            base += "\n\nInclude: Property Description, Encumbrances, Title Exceptions, and Recommended Actions."

        return base

    def _get_user_prompt(self, query: str, context: str, draft_type: str) -> str:
        draft_type_display = draft_type.replace('_', ' ').title()
        return f"""**Task**: Generate a {draft_type_display}

**User Query**: {query}

**Evidence from Documents** (use ONLY this information):
{context}

**Instructions**:
1. Base your response solely on the evidence above
2. Cite specific sources when making factual claims
3. If information is missing, state that clearly
4. Use a professional legal writing style

Generate the requested draft:"""