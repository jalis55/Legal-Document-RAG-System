# src/feedback/feedback_handler.py
"""
Feedback System - Learn from Operator Edits
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


class FeedbackHandler:
    def __init__(self, feedback_dir: str = "data/feedback"):
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_file = self.feedback_dir / "edits_history.jsonl"

    def save_feedback(self, query: str, original_draft: str, 
                     edited_draft: str, user_notes: str = ""):
        """Save operator's edit"""
        feedback = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "original_draft": original_draft,
            "edited_draft": edited_draft,
            "user_notes": user_notes,
            "edit_length_diff": len(edited_draft) - len(original_draft)
        }

        with open(self.feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback) + "\n")

        print("✅ Feedback saved successfully! System can improve from this.")

    def get_recent_feedback(self, limit: int = 5) -> list:
        """Load recent edits for prompt improvement"""
        if not self.feedback_file.exists():
            return []
        
        feedback_list = []
        with open(self.feedback_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    feedback_list.append(json.loads(line.strip()))
                except:
                    continue
        return feedback_list[-limit:]


# Singleton instance
feedback_handler = FeedbackHandler()