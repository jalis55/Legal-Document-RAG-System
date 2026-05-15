"""
Document Processor for Legal-Style Documents
Handles digital PDFs + scanned/messy documents with OCR fallback.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

import pypdf
import pdfplumber
import pdf2image
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter


class DocumentProcessor:
    def __init__(self, output_dir: str = "data/processed", tesseract_cmd: str = None):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # OCR config for legal documents
        self.ocr_config = r'--oem 3 --psm 6 -l eng'

    def process_pdf(self, pdf_path: str, use_ocr_fallback: bool = True) -> Dict:
        """Main method to process a PDF document."""
        pdf_path = Path(pdf_path)
        doc_id = pdf_path.stem.replace(" ", "_")
        
        result = {
            "doc_id": doc_id,
            "filename": pdf_path.name,
            "processed_at": datetime.now().isoformat(),
            "total_pages": 0,
            "pages": [],
            "full_text": "",
            "structured": {},
            "metadata": self._extract_metadata(pdf_path)
        }

        # Step 1: Try native text extraction
        native_texts = self._extract_native_text(pdf_path)
        
        # If native extraction failed, fallback to pdfplumber for page count
        if not native_texts:
            native_texts = self._get_page_count_fallback(pdf_path)
        
        for page_num, native_text in enumerate(native_texts, 1):
            page_data = {
                "page_number": page_num,
                "text": native_text.strip(),
                "ocr_used": False,
                "confidence": 1.0 if len(native_text.strip()) > 100 else 0.6,
                "source": "native"
            }

            # Step 2: OCR fallback for scanned / low-text pages
            if use_ocr_fallback and self._needs_ocr(native_text):
                print(f"   Page {page_num}: OCR needed, processing...")
                ocr_result = self._perform_ocr(pdf_path, page_num)
                page_data.update({
                    "text": ocr_result["text"],
                    "ocr_used": True,
                    "confidence": ocr_result["confidence"],
                    "source": "ocr"
                })

            result["pages"].append(page_data)
            result["full_text"] += f"\n\n--- Page {page_num} ---\n{page_data['text']}"

        result["total_pages"] = len(result["pages"])

        # Step 3: Structured Extraction
        result["structured"] = self._extract_structured_data(pdf_path, result["full_text"])

        # Save processed output with flush
        self._save_json(result, doc_id)
        
        return result

    def _extract_native_text(self, pdf_path: Path) -> List[str]:
        """Extract text using PyPDF (fast for digital PDFs)."""
        texts = []
        try:
            with open(pdf_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    text = page.extract_text() or ""
                    texts.append(text)
            return texts
        except Exception as e:
            print(f"   Native extraction failed: {e}")
            return []

    def _get_page_count_fallback(self, pdf_path: Path) -> List[str]:
        """Get page count as fallback when extraction fails"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                return [""] * len(pdf.pages)
        except Exception:
            return [""] * 10  # Conservative estimate

    def _needs_ocr(self, text: str) -> bool:
        """Heuristic to detect if OCR is needed."""
        if len(text.strip()) < 150:
            return True
        # Count words that are very short (potential OCR garbage)
        words = text.split()
        if len(words) > 0:
            short_word_ratio = sum(1 for w in words if len(w) < 3) / len(words)
            if short_word_ratio > 0.4:
                return True
        return False

    def _perform_ocr(self, pdf_path: Path, page_num: int) -> Dict:
        """Convert PDF page to image and run OCR with preprocessing."""
        try:
            images = pdf2image.convert_from_path(
                pdf_path, 
                first_page=page_num, 
                last_page=page_num,
                dpi=200  # Reduced from 300 for speed
            )
            
            if not images:
                return {"text": "", "confidence": 0.0}
                
            img = images[0]
            
            # Basic image preprocessing for better OCR
            img = img.convert('L')  # grayscale
            img = ImageEnhance.Contrast(img).enhance(2.0)
            img = img.filter(ImageFilter.MedianFilter())
            
            text = pytesseract.image_to_string(img, config=self.ocr_config)
            
            # Get confidence (rough estimate)
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_conf = sum(confidences) / len(confidences) / 100 if confidences else 0.0
            
            return {
                "text": text.strip(),
                "confidence": round(avg_conf, 2)
            }
            
        except Exception as e:
            print(f"   OCR failed on page {page_num}: {e}")
            return {"text": "", "confidence": 0.0}

    def _extract_structured_data(self, pdf_path: Path, full_text: str) -> Dict:
        """Extract tables and basic legal fields."""
        structured = {
            "tables": [],
            "dates": [],
            "parties": [],
            "title": "",
            "key_clauses": []
        }

        # Extract tables using pdfplumber
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables()
                    for table in tables:
                        if table and any(cell for row in table for cell in row if cell):
                            structured["tables"].append(table)
        except Exception as e:
            print(f"   Table extraction skipped: {e}")

        # Simple regex-based extraction
        import re
        date_pattern = r'\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{4}-\d{2}-\d{2})\b'
        structured["dates"] = list(set(re.findall(date_pattern, full_text)))

        # Robust party detection using patterns
        party_keywords = ["between", "party", "agreement made by", "hereinafter", "and"]
        header_text = "\n".join(full_text.split('\n')[:40])
        
        # Look for "BETWEEN [Party A] AND [Party B]"
        between_pattern = r"BETWEEN\s+(.*?)\s+AND\s+(.*?)(?:\.|\n|;)"
        matches = re.search(between_pattern, header_text, re.IGNORECASE | re.DOTALL)
        if matches:
            structured["parties"].extend([matches.group(1).strip(), matches.group(2).strip()])
        else:
            # Fallback to keyword-based line detection
            for line in header_text.split('\n'):
                if any(kw.lower() in line.lower() for kw in party_keywords):
                    if len(line.strip()) > 5 and len(line.strip()) < 200:
                        structured["parties"].append(line.strip())
        
        # Deduplicate and clean
        structured["parties"] = list(set([p for p in structured["parties"] if len(p) > 2]))
        
        # Try to extract title (first line that looks like a title)
        first_lines = full_text.split('\n')[:10]
        for line in first_lines:
            if len(line.strip()) > 10 and len(line.strip()) < 100:
                structured["title"] = line.strip()
                break

        return structured

    def _extract_metadata(self, pdf_path: Path) -> Dict:
        return {
            "file_size_bytes": os.path.getsize(pdf_path),
            "created_at": datetime.fromtimestamp(os.path.getctime(pdf_path)).isoformat(),
            "modified_at": datetime.fromtimestamp(os.path.getmtime(pdf_path)).isoformat()
        }

    def _save_json(self, data: Dict, doc_id: str):
        output_path = self.output_dir / f"{doc_id}_processed.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        
        print(f"✅ Processed: {output_path}")


# For quick testing
if __name__ == "__main__":
    processor = DocumentProcessor()
    # Example usage:
    # result = processor.process_pdf("data/raw/sample_contract.pdf")
    # print(f"Extracted {len(result['pages'])} pages | OCR used: {any(p['ocr_used'] for p in result['pages'])}")