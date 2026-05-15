# Legal Document RAG System - Pearson Specter Litt AI Engineer Assessment

## 🚀 Overview
A robust, grounded Retrieval-Augmented Generation (RAG) pipeline designed for Pearson Specter Litt. This system ingests "messy" legal documents, extracts structured data, and generates professional legal drafts that improve over time by learning from operator edits.

## ✨ Key Features
- **Intelligent Document Processing**: OCR fallback for scanned/low-quality PDFs with automated image enhancement (Contrast, Grayscale, Median Filter).
- **Grounded Retrieval**: FAISS-based semantic search using `sentence-transformers` for high-precision context retrieval.
- **Smart Feedback Loop**: Few-shot learning implementation where the system learns from previous operator edits to improve tone, style, and accuracy.
- **Duplicate Handling**: Intelligent skip logic in the ingestion pipeline to prevent redundant data.
- **Atomic Operations**: Database and file updates use atomic renames to prevent corruption.
- **Multiple Draft Types**: Specialized support for Case Fact Summaries, Title Reviews, Document Checklists, Internal Memos, and more.

---

## 🏗️ Architecture Overview
The system is built with a modular, scalable architecture:

1.  **Processing Layer (`src/processing`)**:
    *   Uses `pypdf` for digital text extraction.
    *   Uses `pdf2image` + `pytesseract` for OCR on scanned pages.
    *   Uses `pdfplumber` for robust table extraction.
    *   Extracts structured metadata (Parties, Dates, Titles).

2.  **Retrieval Layer (`src/retrieval`)**:
    *   Chunks text with configurable overlap to preserve context.
    *   Generates embeddings using `all-MiniLM-L6-v2`.
    *   Stores vectors in a FAISS index for sub-millisecond retrieval.

3.  **Generation Layer (`src/generation`)**:
    *   Powered by Groq (Llama3/Qwen) for ultra-fast, high-quality drafting.
    *   Implements a system prompt that strictly enforces grounding (evidence-only).
    *   Injects operator feedback as few-shot examples to align with firm-specific style.

4.  **Feedback Layer (`src/feedback`)**:
    *   Captures user edits and notes.
    *   Computes edit diffs and stores historical corrections for the learning loop.

---

## 🛠️ Setup & Installation

### Prerequisites
*   Python 3.10+
*   Tesseract OCR engine (`sudo apt install tesseract-ocr` on Linux)
*   Poppler (for PDF processing: `sudo apt install poppler-utils` on Linux)

### Installation
1.  Clone the repository and navigate to the directory.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Configure environment variables:
    Create a `.env` file in the root directory:
    ```env
    GROQ_API_KEY=your_key_here
    MODEL_NAME=qwen/qwen3-32b
    ```

### Running with Docker (Optional)
If you prefer to run the system in a container:
1.  Build the image:
    ```bash
    docker build -t legal-rag-system .
    ```
2.  Run the container:
    ```bash
    docker run -p 8501:8501 --env-file .env legal-rag-system
    ```

---

## 🚦 Project Workflow

### 1. Ingest Documents
Place your PDFs in `data/raw/` and run the main pipeline:
```bash
python main.py
```
This will process the PDFs, perform OCR if needed, and index them into the vector database.

### 2. Generate Drafts & Provide Feedback
You have two ways to interact with the system:

#### A. Modern Web Interface (Recommended)
Run the Streamlit app for a premium experience with document uploads and visual source tracking:
```bash
streamlit run app.py
```

#### B. Terminal-based Generator
Run the interactive CLI generator:
```bash
python query_test.py
```
You can select a draft type, enter your query, review the grounded output (with sources), and provide an edited version to teach the system.

### 3. Rebuild or Update Index
If you want to manually manage the vector database:
```bash
python ingest_to_vector.py
```
Choose between **Update** (only index new files) or **Rebuild** (clear and re-index all).

---

## 📊 Evaluation Approach
*   **Retrieval Quality**: Verified using `evaluate_system.py` which measures `relevance_score` and checks source alignment.
    ```bash
    python evaluate_system.py
    ```
*   **Grounding**: Strictly enforced via system prompts that prohibit hallucinations and require source citations.
*   **Improvement Loop**: Quantified by the inclusion of operator edits in future prompt context, allowing the model to bridge the gap between "default" and "firm-preferred" output.

## 📝 Assumptions & Trade-offs
*   **Local Embeddings**: Chose `all-MiniLM-L6-v2` for local performance and speed; can be upgraded to larger models for higher semantic depth.
*   **FAISS IndexFlatL2**: Chosen for maximum accuracy on the current dataset size; for massive scaling (1M+ documents), an IVF or HNSW index would be substituted.
*   **Few-shot feedback**: Limited to the last 3 examples to keep the prompt concise and relevant to the most recent operator preferences.