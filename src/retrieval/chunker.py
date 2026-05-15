# src/retrieval/chunker.py
import json
from pathlib import Path
from typing import Dict, List


class DocumentChunker:
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_processed_document(self, processed_json_path: str) -> List[Dict]:
        """Take processed JSON and return clean chunks."""
        with open(processed_json_path, 'r', encoding='utf-8') as f:
            doc = json.load(f)

        all_chunks = []
        
        for page in doc.get("pages", []):
            page_text = page.get("text", "")
            page_num = page.get("page_number")
            
            if not page_text.strip():
                continue
                
            page_chunks = self._split_text_into_chunks(
                text=page_text,
                page_number=page_num,
                doc_id=doc["doc_id"],
                filename=doc["filename"]
            )
            all_chunks.extend(page_chunks)

        # Also add some metadata chunks (title, structured data)
        if doc.get("structured"):
            meta_text = f"Document Title: {doc['structured'].get('title', '')}\n"
            meta_text += f"Parties: {', '.join(doc['structured'].get('parties', []))}\n"
            meta_text += f"Dates: {', '.join(doc['structured'].get('dates', []))}"
            
            all_chunks.append({
                "chunk_id": f"{doc['doc_id']}_metadata",
                "doc_id": doc["doc_id"],
                "filename": doc["filename"],
                "page_number": 0,
                "text": meta_text,
                "type": "metadata"
            })

        print(f"Created {len(all_chunks)} chunks from {doc['filename']}")
        return all_chunks

    def _split_text_into_chunks(self, text: str, page_number: int, doc_id: str, filename: str) -> List[Dict]:
        """Simple but effective overlapping chunker."""
        words = text.split()
        chunks = []
        step = self.chunk_size - self.chunk_overlap

        for i in range(0, len(words), max(step, 1)):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            if len(chunk_text.split()) < 50:  # skip very small chunks
                continue

            chunks.append({
                "chunk_id": f"{doc_id}_p{page_number}_c{len(chunks)}",
                "doc_id": doc_id,
                "filename": filename,
                "page_number": page_number,
                "text": chunk_text,
                "token_count": len(chunk_words),
                "type": "content"
            })
        
        return chunks


# Quick test
if __name__ == "__main__":
    chunker = DocumentChunker(chunk_size=700, chunk_overlap=150)
    # Example:
    # chunks = chunker.chunk_processed_document("data/processed/your_file_processed.json")
    # print(f"Total chunks: {len(chunks)}")