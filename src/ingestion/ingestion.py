# src/ingestion/ingestion.py
from pathlib import Path
from typing import List

from src.retrieval.chunker import DocumentChunker
from src.retrieval.retriever import LegalRetriever


class DocumentIngestionPipeline:
    def __init__(self):
        self.chunker = DocumentChunker()
        self.retriever = LegalRetriever()

    def ingest_processed_file(self, processed_json_path: str, skip_if_exists: bool = True):
        """Chunk + Add one processed document to vector DB"""
        path = Path(processed_json_path)
        if not path.exists():
            print(f"❌ File not found: {path}")
            return False

        doc_id = path.stem.replace("_processed", "")
        if skip_if_exists and doc_id in self.retriever.get_indexed_doc_ids():
            print(f"   ⏩ Skipping {path.name} (already indexed)")
            return True

        print(f"   📄 Ingesting {path.name}...")
        
        chunks = self.chunker.chunk_processed_document(str(path))
        
        if chunks:
            self.retriever.add_chunks(chunks)
            return True
        return False

    def ingest_all(self, processed_dir: str = "data/processed", skip_if_exists: bool = True):
        """Ingest all processed JSON files"""
        processed_dir = Path(processed_dir)
        
        if not processed_dir.exists():
            print(f"❌ Directory not found: {processed_dir}")
            print("   Please run 'python main.py' to process documents first.")
            return
        
        files = list(processed_dir.glob("*_processed.json"))
        
        if not files:
            print(f"⚠️  No processed JSON files found in {processed_dir}")
            print("   Please run 'python main.py' to process documents first.")
            return
        
        print(f"📁 Found {len(files)} processed file(s)")
        
        successful = 0
        for i, file in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] ", end="")
            if self.ingest_processed_file(str(file), skip_if_exists=skip_if_exists):
                successful += 1

        print(f"\n{'='*50}")
        print(f"✅ Ingestion completed: {successful}/{len(files)} files processed")
        print(f"{'='*50}")
        self.retriever.get_stats()