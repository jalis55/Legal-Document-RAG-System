# query_test.py
"""
Interactive Grounded Draft Generator with Feedback Loop
"""

from src.generation.drafter import LegalDrafter
from src.feedback.feedback_handler import feedback_handler


def main():
    print("=" * 85)
    print("🔍 Legal RAG System - Draft Generator + Feedback Loop")
    print("=" * 85)
    
    try:
        drafter = LegalDrafter()
    except ValueError as e:
        print(f"❌ Cannot initialize: {e}")
        print("Please ensure GROQ_API_KEY is set in .env file")
        return

    # Check if vector DB has content
    stats = drafter.retriever.get_stats()
    if stats == 0:
        print("\n⚠️  Warning: Vector database is empty!")
        print("   Please run 'python main.py' first to ingest documents.")
        print("   Or run 'python ingest_to_vector.py' if you have processed files.\n")
        
        proceed = input("Continue anyway? (y/n): ").strip().lower()
        if proceed != 'y':
            return

    while True:
        try:
            print("\n" + "-"*70)
            print("Draft Types:")
            print("1. Case Fact Summary    2. Title Review     3. Notice Summary")
            print("4. Document Checklist   5. Internal Memo    6. General")
            print("-"*70)

            choice = input("\nSelect draft type (1-6) or 'exit': ").strip()
            if choice.lower() in ['exit', 'quit', 'q']:
                break

            draft_map = {
                "1": "case_fact_summary", 
                "2": "title_review_summary", 
                "3": "notice_related_summary", 
                "4": "document_checklist",
                "5": "first_pass_internal_memo", 
                "6": "general_analysis"
            }
            
            draft_type = draft_map.get(choice, "general_analysis")
            query = input("\nEnter query: ").strip()

            if not query:
                print("⚠️  Query cannot be empty.")
                continue

            print("\n📝 Generating draft...")
            result = drafter.generate_draft(query=query, draft_type=draft_type)

            if "error" in result:
                print(f"❌ Error: {result['error']}")
                continue

            # Show sources used
            print("\n" + "="*80)
            print("📄 GENERATED DRAFT")
            print("="*80)
            print(result["draft"])
            print("\n" + "="*80)
            print("📚 SOURCES USED:")
            for i, source in enumerate(result["sources"], 1):
                print(f"   {i}. {source['filename']} (Page {source['page_number']}, "
                      f"Relevance: {source['relevance_score']})")
            print("="*80)

            # === Feedback Collection ===
            print("\n💡 Would you like to edit this draft? (y/n)")
            if input().lower() == 'y':
                print("\n📝 Paste your edited version below (type 'END' on new line when done):")
                lines = []
                while True:
                    line = input()
                    if line == "END":
                        break
                    lines.append(line)
                edited_draft = "\n".join(lines)

                notes = input("\n📝 Any notes about your changes? (optional): ").strip()

                # Save feedback
                feedback_handler.save_feedback(
                    query=query,
                    original_draft=result["draft"],
                    edited_draft=edited_draft,
                    user_notes=notes
                )
                
                print("\n✨ Thank you! Your feedback will help improve future drafts.")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()