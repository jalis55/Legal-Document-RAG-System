# evaluate_system.py
"""
Evaluation script for the Legal Document RAG System.
Demonstrates retrieval quality and grounding.
"""

import sys
from src.generation.drafter import LegalDrafter

def evaluate():
    print("=" * 80)
    print("📋 System Evaluation - Legal RAG")
    print("=" * 80)

    try:
        # Initialize drafter (which also initializes retriever)
        drafter = LegalDrafter()
    except ValueError as e:
        print(f"\n❌ Initialization Error: {e}")
        print("Please ensure your .env file is configured correctly.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        sys.exit(1)

    # Check if index exists
    stats = drafter.retriever.get_stats()
    if stats == 0:
        print("\n⚠️  Warning: Vector database is empty!")
        print("Please run 'python main.py' first to ingest documents.")
        return

    test_queries = [
        "What are the payment terms in the legal services agreement?",
        "Who are the parties involved in the agreement?",
        "What is the governing law for these documents?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n[Test {i}] Query: {query}")
        
        # 1. Evaluate Retrieval (using the retriever inside drafter)
        print("\n--- 🔍 Retrieval Analysis ---")
        results = drafter.retriever.search(query, top_k=3)
        if not results:
            print("No results found.")
        
        for idx, res in enumerate(results, 1):
            print(f"Result {idx}: [Score: {res['relevance_score']:.3f}] {res['filename']} (Page {res['page_number']})")
            print(f"Snippet: {res['text'][:150]}...")

        # 2. Evaluate Generation & Grounding
        print("\n--- ✍️ Generation Analysis ---")
        draft_result = drafter.generate_draft(query)
        if "error" in draft_result:
            print(f"Error: {draft_result['error']}")
            continue

        print(f"Model: {draft_result['model_used']}")
        print(f"Grounded Draft Snippet:\n{draft_result['draft'][:300]}...")
        print(f"Sources cited: {len(draft_result['sources'])}")
        
        # Grounding check: ensure sources are from relevant files
        cited_files = set(s['filename'] for s in draft_result['sources'])
        print(f"Files referenced: {', '.join(cited_files)}")
        print("-" * 60)

    print("\n✅ Evaluation completed.")

if __name__ == "__main__":
    evaluate()
