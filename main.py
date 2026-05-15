# main.py
"""
Legal Document RAG System - Main Pipeline
Pearson Specter Litt AI Engineer Assessment
"""

from pathlib import Path
import sys
import time
from dotenv import load_dotenv

# Load environment variables (Groq API key)
load_dotenv()

from src.processing.extractor import DocumentProcessor
from src.ingestion.ingestion import DocumentIngestionPipeline
from src.generation.drafter import LegalDrafter


def main():
    print("=" * 85)
    print("🚀 Pearson Specter Litt - Legal Document RAG System")
    print("=" * 85)

    # Initialize components
    processor = DocumentProcessor()
    ingestion_pipeline = DocumentIngestionPipeline()
    
    # Check for API key early
    try:
        drafter = LegalDrafter()
    except ValueError as e:
        print(f"⚠️  {e}")
        print("Please set GROQ_API_KEY in .env file")
        drafter = None

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(raw_dir.glob("*.pdf"))

    print(f"\n📁 Found {len(pdf_files)} PDF file(s) in data/raw/\n")

    if pdf_files:
        print("📑 Stage 1: Processing Documents (OCR + Structure)")
        print("-" * 60)
        
        processed_files = []
        for pdf_file in pdf_files:
            try:
                print(f"Processing → {pdf_file.name}")
                result = processor.process_pdf(str(pdf_file))
                print(f"   ✅ Completed ({result['total_pages']} pages, OCR used: {any(p['ocr_used'] for p in result['pages'])})")
                processed_files.append(result['doc_id'])
                
                # Small delay to ensure file is written
                time.sleep(0.1)
                
            except Exception as e:
                print(f"   ❌ Failed {pdf_file.name}: {e}")

        # Stage 2: Ingestion
        if processed_files:
            print("\n🔍 Stage 2: Chunking + Indexing into FAISS")
            print("-" * 60)
            ingestion_pipeline.ingest_all()
        else:
            print("\n⚠️  No files were successfully processed.")

    else:
        print("ℹ️  No PDFs found in data/raw/")
        # Check if there are existing processed files to ingest
        existing_files = list(processed_dir.glob("*_processed.json"))
        if existing_files:
            print(f"   Found {len(existing_files)} existing processed files.")
            print("\n🔍 Stage 2: Indexing existing files into FAISS")
            print("-" * 60)
            ingestion_pipeline.ingest_all()
        else:
            print("   No existing processed files found either.")

    # Final Status
    print("\n" + "=" * 85)
    print("🎉 System is Ready!")
    print("=" * 85)
    print("Available Commands:")
    print("   • python query_test.py          → Interactive draft generator")
    print("   • python ingest_to_vector.py    → Rebuild vector DB only")
    print("   • python -m src.ingestion.ingestion → Ingest only")
    print("=" * 85)


if __name__ == "__main__":
    main()