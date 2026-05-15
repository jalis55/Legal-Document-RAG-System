# ingest_to_vector.py
"""
Standalone script to rebuild vector database from processed files
Run this after processing PDFs or when you want to reset the vector DB
"""

import sys
from pathlib import Path
from src.ingestion.ingestion import DocumentIngestionPipeline


def main():
    print("=" * 60)
    print("🔄 Vector Database Rebuilder")
    print("=" * 60)
    
    processed_dir = Path("data/processed")
    
    if not processed_dir.exists():
        print("❌ No processed files found. Run 'python main.py' first.")
        return
    
    files = list(processed_dir.glob("*_processed.json"))
    
    if not files:
        print("❌ No processed JSON files found in data/processed/")
        print("   Please run 'python main.py' to process PDFs first.")
        return
    
    print(f"Found {len(files)} processed file(s):")
    for f in files:
        print(f"   • {f.name}")
    
    print("\nSelect Mode:")
    print("1. Update (Skip already indexed files)")
    print("2. Rebuild (Clear all and re-index everything)")
    
    mode = input("\nSelect mode (1 or 2): ").strip()
    
    pipeline = DocumentIngestionPipeline()
    
    if mode == "2":
        print("\n⚠️  Clearing existing vector database...")
        pipeline.retriever.clear_index()
        skip = False
    else:
        print("\n🔍 Running in Update mode (skipping duplicates)...")
        skip = True
    
    print("\n🚀 Processing documents...")
    pipeline.ingest_all(skip_if_exists=skip)
    
    print("\n✅ Done!")
    print("   You can now run 'python query_test.py' to generate drafts.")


if __name__ == "__main__":
    main()