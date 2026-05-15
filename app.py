# app.py
"""
Legal Document RAG System - Streamlit Interface
Pearson Specter Litt AI Engineer Assessment
"""

import streamlit as st
import os
from pathlib import Path
import tempfile
import time
from datetime import datetime
import pandas as pd

# Import project modules
from src.processing.extractor import DocumentProcessor
from src.ingestion.ingestion import DocumentIngestionPipeline
from src.generation.drafter import LegalDrafter
from src.feedback.feedback_handler import feedback_handler

# Page configuration
st.set_page_config(
    page_title="Legal Document RAG System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .source-card {
        background-color: #f0f2f6;
        padding: 0.75rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }
    .feedback-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #00a3ff;
        margin: 1rem 0;
    }
    .stat-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .draft-box {
        background-color: #faf9f6;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)


class StreamlitRAGSystem:
    def __init__(self):
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'processor' not in st.session_state:
            st.session_state.processor = DocumentProcessor()
        if 'ingestion_pipeline' not in st.session_state:
            st.session_state.ingestion_pipeline = DocumentIngestionPipeline()
        if 'drafter' not in st.session_state:
            try:
                st.session_state.drafter = LegalDrafter()
            except ValueError as e:
                st.session_state.drafter = None
                st.session_state.drafter_error = str(e)
        if 'current_draft' not in st.session_state:
            st.session_state.current_draft = None
        if 'feedback_count' not in st.session_state:
            st.session_state.feedback_count = 0
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = []
            
    def render_sidebar(self):
        """Render sidebar with system info and controls"""
        with st.sidebar:
            st.image("https://img.icons8.com/color/96/000000/law.png", width=80)
            st.markdown("## ⚖️ Pearson Specter Litt")
            st.markdown("*AI-Powered Legal Document System*")
            st.divider()
            
            # System Status
            st.markdown("### 📊 System Status")
            
            col1, col2 = st.columns(2)
            with col1:
                # Check vector DB status
                if st.session_state.drafter:
                    stats = st.session_state.drafter.retriever.get_stats()
                    st.metric("📚 Chunks Indexed", stats)
                else:
                    st.metric("📚 Chunks Indexed", "N/A")
                    
            with col2:
                # Check feedback count
                feedback_files = list(Path("data/feedback").glob("*.jsonl")) if Path("data/feedback").exists() else []
                feedback_total = 0
                for f in feedback_files:
                    with open(f) as fp:
                        feedback_total += sum(1 for _ in fp)
                st.metric("💬 Feedback Entries", feedback_total)
            
            st.divider()
            
            # Quick Actions
            st.markdown("### 🚀 Quick Actions")
            if st.button("🔄 Rebuild Vector DB", use_container_width=True):
                with st.spinner("Rebuilding vector database..."):
                    st.session_state.ingestion_pipeline.ingest_all()
                st.success("Vector DB rebuilt!")
                st.rerun()
            
            if st.button("🗑️ Clear All Feedback", use_container_width=True):
                feedback_dir = Path("data/feedback")
                if feedback_dir.exists():
                    for f in feedback_dir.glob("*.jsonl"):
                        f.unlink()
                st.success("Feedback cleared!")
                st.rerun()
            
            st.divider()
            
            # Document Stats
            st.markdown("### 📄 Document Status")
            processed_dir = Path("data/processed")
            if processed_dir.exists():
                processed_files = list(processed_dir.glob("*_processed.json"))
                st.write(f"**Processed Files:** {len(processed_files)}")
                for f in processed_files[:5]:
                    st.caption(f"• {f.name.replace('_processed.json', '')}")
                if len(processed_files) > 5:
                    st.caption(f"... and {len(processed_files)-5} more")
            else:
                st.info("No processed documents yet")
            
            st.divider()
            
            # Instructions
            with st.expander("📖 How to Use"):
                st.markdown("""
                1. **Upload Documents** → Process PDFs with OCR
                2. **View Documents** → See extracted content
                3. **Generate Drafts** → Ask questions to create legal drafts
                4. **Provide Feedback** → Edit drafts to improve the system
                """)
            
            # Environment check
            st.divider()
            if st.session_state.drafter:
                st.success("✅ Groq API Connected")
            else:
                st.error("❌ Groq API Not Configured")
                if hasattr(st.session_state, 'drafter_error'):
                    st.caption(f"Error: {st.session_state.drafter_error}")
                st.info("Add GROQ_API_KEY to .env file")
    
    def render_upload_tab(self):
        """Document upload and processing tab"""
        st.markdown("### 📤 Upload Legal Documents")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_files = st.file_uploader(
                "Upload PDF documents",
                type=['pdf'],
                accept_multiple_files=True,
                help="Upload scanned or digital PDFs. System will automatically apply OCR if needed."
            )
            
        with col2:
            st.markdown("#### ⚙️ Processing Options")
            use_ocr = st.checkbox("Enable OCR Fallback", value=True)
            auto_ingest = st.checkbox("Auto-ingest after processing", value=True)
        
        if uploaded_files:
            st.markdown("---")
            st.markdown("#### 📑 Processing Queue")
            
            for uploaded_file in uploaded_files:
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"**{uploaded_file.name}**")
                with col2:
                    file_size = len(uploaded_file.getvalue()) / 1024
                    st.write(f"{file_size:.1f} KB")
                with col3:
                    if st.button(f"Process", key=f"process_{uploaded_file.name}"):
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Process the PDF
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            try:
                                result = st.session_state.processor.process_pdf(
                                    tmp_path, 
                                    use_ocr_fallback=use_ocr
                                )
                                
                                st.success(f"✅ Processed {uploaded_file.name}")
                                st.info(f"📄 {result['total_pages']} pages | OCR used: {any(p['ocr_used'] for p in result['pages'])}")
                                
                                # Auto-ingest if enabled
                                if auto_ingest:
                                    with st.spinner("Ingesting into vector database..."):
                                        json_path = Path(f"data/processed/{result['doc_id']}_processed.json")
                                        if json_path.exists():
                                            st.session_state.ingestion_pipeline.ingest_processed_file(str(json_path))
                                            st.success("✅ Ingested to vector DB")
                                
                                # Cleanup
                                os.unlink(tmp_path)
                                
                            except Exception as e:
                                st.error(f"❌ Failed: {str(e)}")
    
    def render_documents_tab(self):
        """View processed documents tab"""
        st.markdown("### 📚 Processed Documents")
        
        processed_dir = Path("data/processed")
        if not processed_dir.exists():
            st.info("No processed documents yet. Upload and process some PDFs first.")
            return
        
        processed_files = list(processed_dir.glob("*_processed.json"))
        
        if not processed_files:
            st.info("No processed documents found.")
            return
        
        # Document selector
        doc_names = [f.name.replace('_processed.json', '') for f in processed_files]
        selected_doc = st.selectbox("Select a document to view:", doc_names)
        
        if selected_doc:
            doc_path = processed_dir / f"{selected_doc}_processed.json"
            import json
            with open(doc_path, 'r', encoding='utf-8') as f:
                doc_data = json.load(f)
            
            # Document info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Pages", doc_data['total_pages'])
            with col2:
                ocr_pages = sum(1 for p in doc_data['pages'] if p.get('ocr_used', False))
                st.metric("OCR Pages", ocr_pages)
            with col3:
                st.metric("File Size", f"{doc_data['metadata']['file_size_bytes'] / 1024:.1f} KB")
            with col4:
                st.metric("Parties Found", len(doc_data['structured'].get('parties', [])))
            
            # Structured data
            with st.expander("📋 Extracted Structured Data"):
                if doc_data['structured'].get('parties'):
                    st.markdown("**Parties:**")
                    for party in doc_data['structured']['parties']:
                        st.write(f"• {party}")
                
                if doc_data['structured'].get('dates'):
                    st.markdown("**Dates:**")
                    for date in doc_data['structured']['dates'][:10]:
                        st.write(f"• {date}")
                
                if doc_data['structured'].get('tables'):
                    st.markdown(f"**Tables:** {len(doc_data['structured']['tables'])} tables extracted")
            
            # Page content viewer
            page_num = st.slider("View Page:", 1, doc_data['total_pages'], 1)
            page_data = doc_data['pages'][page_num - 1]
            
            st.markdown(f"#### Page {page_num}")
            if page_data.get('ocr_used', False):
                st.caption(f"🔍 OCR Used (Confidence: {page_data.get('confidence', 0)}))")
            
            with st.container(height=400):
                st.text(page_data['text'][:5000])
    
    def render_generate_tab(self):
        """Draft generation tab"""
        st.markdown("### ✍️ Generate Legal Draft")
        
        # Check if drafter is available
        if not st.session_state.drafter:
            st.error("Cannot generate drafts: Groq API not configured")
            st.info("Please add GROQ_API_KEY to your .env file and restart the app")
            return
        
        # Check if vector DB has content
        if st.session_state.drafter.retriever.get_stats() == 0:
            st.warning("⚠️ Vector database is empty. Please upload and process documents first.")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            draft_type = st.selectbox(
                "Draft Type:",
                options=[
                    "general_analysis",
                    "case_fact_summary",
                    "title_review_summary",
                    "notice_related_summary",
                    "document_checklist",
                    "first_pass_internal_memo"
                ],
                format_func=lambda x: x.replace('_', ' ').title()
            )
        
        with col2:
            top_k = st.slider("Number of sources to retrieve:", 3, 10, 6)
        
        query = st.text_area(
            "Enter your query:",
            placeholder="Example: Summarize the key parties and their obligations in the contract...",
            height=100
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            generate_button = st.button("🚀 Generate Draft", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_button:
            st.session_state.current_draft = None
            st.rerun()
        
        if generate_button and query:
            with st.spinner("Generating draft..."):
                result = st.session_state.drafter.generate_draft(
                    query=query,
                    top_k=top_k,
                    draft_type=draft_type
                )
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.session_state.current_draft = result
                    st.session_state.current_query = query
        
        # Display draft if exists
        if st.session_state.get('current_draft'):
            result = st.session_state.current_draft
            
            # Sources section
            with st.expander("📚 Sources Used", expanded=False):
                for i, source in enumerate(result['sources'], 1):
                    st.markdown(f"""
                    <div class="source-card">
                        <strong>Source {i}</strong>: {source['filename']} (Page {source['page_number']})<br>
                        <strong>Relevance Score</strong>: {source['relevance_score']}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Draft content
            st.markdown("#### 📝 Generated Draft")
            
            # Editable draft
            edited_draft = st.text_area(
                "Edit the draft if needed:",
                value=result['draft'],
                height=400,
                key="draft_editor"
            )
            
            # Feedback section
            st.markdown("---")
            st.markdown("#### 💬 Provide Feedback")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                submit_feedback = st.button("✅ Submit Feedback & Save", use_container_width=True)
            with col2:
                st.caption("Your edits will help improve future drafts")
            
            if submit_feedback:
                if edited_draft != result['draft']:
                    notes = st.text_area("Any notes about your changes? (optional)", key="feedback_notes")
                    
                    feedback_handler.save_feedback(
                        query=st.session_state.current_query,
                        original_draft=result['draft'],
                        edited_draft=edited_draft,
                        user_notes=notes
                    )
                    
                    st.session_state.feedback_count += 1
                    st.success("✅ Feedback saved! The system will learn from your edits.")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.info("No changes detected. Edit the draft before submitting feedback.")
    
    def render_feedback_tab(self):
        """View feedback history tab"""
        st.markdown("### 📊 Feedback History")
        
        feedback_dir = Path("data/feedback")
        if not feedback_dir.exists():
            st.info("No feedback data yet. Generate some drafts and provide feedback!")
            return
        
        feedback_files = list(feedback_dir.glob("*.jsonl"))
        if not feedback_files:
            st.info("No feedback entries found.")
            return
        
        # Load all feedback
        all_feedback = []
        for feedback_file in feedback_files:
            with open(feedback_file, 'r', encoding='utf-8') as f:
                for line in f:
                    import json
                    try:
                        all_feedback.append(json.loads(line.strip()))
                    except:
                        pass
        
        if not all_feedback:
            st.info("No feedback entries found.")
            return
        
        # Display statistics
        st.markdown("#### 📈 Feedback Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Edits", len(all_feedback))
        with col2:
            avg_edit = sum(f.get('edit_length_diff', 0) for f in all_feedback) / len(all_feedback)
            st.metric("Avg Edit Length Change", f"{avg_edit:.0f} chars")
        with col3:
            unique_queries = len(set(f.get('query', '') for f in all_feedback))
            st.metric("Unique Queries", unique_queries)
        
        # Display feedback table
        st.markdown("#### 📝 Recent Feedback")
        
        feedback_df = pd.DataFrame(all_feedback[::-1])  # Reverse for most recent first
        feedback_df['timestamp'] = pd.to_datetime(feedback_df['timestamp'])
        feedback_df['edit_size'] = feedback_df['edit_length_diff'].abs()
        
        # Select columns to display
        display_cols = ['timestamp', 'query', 'edit_size']
        if all(c in feedback_df.columns for c in display_cols):
            st.dataframe(
                feedback_df[display_cols].head(20),
                use_container_width=True,
                column_config={
                    "timestamp": "Date",
                    "query": "Query",
                    "edit_size": "Edit Size (chars)"
                }
            )
        
        # View individual feedback
        st.markdown("#### 🔍 View Individual Feedback")
        selected_idx = st.selectbox(
            "Select a feedback entry to view:",
            range(len(all_feedback)),
            format_func=lambda x: f"{all_feedback[x]['timestamp'][:19]} - {all_feedback[x]['query'][:50]}..."
        )
        
        if selected_idx is not None:
            fb = all_feedback[selected_idx]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Draft:**")
                st.text_area("", fb['original_draft'], height=200, key="orig_view", disabled=True)
            with col2:
                st.markdown("**Edited Draft:**")
                st.text_area("", fb['edited_draft'], height=200, key="edit_view", disabled=True)
            
            if fb.get('user_notes'):
                st.info(f"📝 Notes: {fb['user_notes']}")
    
    def run(self):
        """Main app runner"""
        # Header
        st.markdown("""
        <div class="main-header">
            <h1 style="color: white; margin: 0;">⚖️ Legal Document RAG System</h1>
            <p style="color: #ccc; margin: 0;">Pearson Specter Litt - AI-Powered Legal Drafting Assistant</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar
        self.render_sidebar()
        
        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📤 Upload Documents",
            "📚 View Documents",
            "✍️ Generate Drafts",
            "📊 Feedback History"
        ])
        
        with tab1:
            self.render_upload_tab()
        
        with tab2:
            self.render_documents_tab()
        
        with tab3:
            self.render_generate_tab()
        
        with tab4:
            self.render_feedback_tab()


# Run the app
if __name__ == "__main__":
    app = StreamlitRAGSystem()
    app.run()