"""Stage PDF provisions for human verification before promoting them to the index."""
import argparse
import hashlib
import json
import re
import subprocess
from bisect import bisect_right
from collections import Counter
from io import BytesIO
from pathlib import Path

def extract(path, source_url, ocr=None, instrument=None, jurisdiction="Unspecified", rapid_ocr=False):
    from pypdf import PdfReader
    pages = [(i, (page.extract_text() or "").replace("\u00ad", ""))
             for i, page in enumerate(PdfReader(path).pages, 1)]
    ocr_pages = set()
    if (ocr or rapid_ocr) and any(not text.strip() for _, text in pages):
        import pypdfium2
        recognizer = None
        if rapid_ocr:
            from rapidocr_onnxruntime import RapidOCR
            recognizer = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)
        rendered = pypdfium2.PdfDocument(str(path))
        try:
            for offset, (page_number, text) in enumerate(pages):
                if text.strip():
                    continue
                buffer = BytesIO()
                rendered[offset].render(scale=2.5).to_pil().save(buffer, format="PNG")
                if recognizer:
                    lines, _ = recognizer(buffer.getvalue())
                    text = "\n".join(line[1] for line in (lines or []))
                else:
                    result = subprocess.run([ocr, "stdin", "stdout"], input=buffer.getvalue(),
                                            capture_output=True, check=True, timeout=90)
                    text = result.stdout.decode("utf-8")
                pages[offset] = page_number, text
                ocr_pages.add(page_number)
        finally:
            rendered.close()
    yield from chunk_pages(pages, source_url, instrument or path.stem, jurisdiction, ocr_pages)

def chunk_pages(pages, source_url, instrument, jurisdiction, ocr_pages):
    spacing = r"\s*" if ocr_pages else r"\s+"
    pattern = re.compile(r"(?m)^\s*((?:Rule |Section )?\d+[A-Z]?)\." + spacing + r"([A-Z][^\n]{3,140})")
    combined = "\n".join(text for _, text in pages)
    offsets = []
    pos = 0
    for page, text in pages:
        offsets.append((pos, page))
        pos += len(text) + 1
    positions = [offset for offset, _ in offsets]
    matches = list(pattern.finditer(combined))
    reference_counts = Counter(match[1] for match in matches)
    chapters = list(re.finditer(r"(?mi)^\s*(CHAPTER\s*[IVXLCDM]+)\b[^\r\n]*", combined))
    chapter_starts = [match.start() for match in chapters]
    seen = set()
    for i, match in enumerate(matches):
        end = matches[i+1].start() if i+1 < len(matches) else len(combined)
        original = " ".join(combined[match.start():end].split())
        if not original or original in seen:
            continue
        seen.add(original)
        chunk_id = hashlib.sha256((source_url + original).encode()).hexdigest()[:24]
        chapter_index = bisect_right(chapter_starts, match.start()) - 1
        start_page = offsets[bisect_right(positions, match.start()) - 1][1]
        end_page = offsets[bisect_right(positions, end - 1) - 1][1]
        used_ocr = sorted(page for page in ocr_pages if start_page <= page <= end_page)
        words = len(original.split())
        flags = []
        if reference_counts[match[1]] > 1:
            flags.append("repeated_reference_review")
        if chapter_index < 0:
            flags.append("chapter_unresolved")
        if used_ocr:
            flags.append("compare_ocr_with_source")
        if words > 200:
            flags.append("long_provision_review")
        yield {"chunk_id": chunk_id, "regulation": instrument,
               "chapter_candidate": re.sub(r"(?i)^chapter\s*", "CHAPTER ", chapters[chapter_index][1]) if chapter_index >= 0 else None,
               "provision_reference_candidate": match[1],
               "start_page": start_page,
               "end_page": end_page,
               "ocr_pages": used_ocr, "quality_flags": flags,
               "jurisdiction": jurisdiction, "hazard_tags": [], "applicability": None,
               "word_count": words, "text": original,
               "source_url": source_url, "verification_status": "provisional"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("source_url")
    parser.add_argument("output", type=Path)
    ocr_mode = parser.add_mutually_exclusive_group()
    ocr_mode.add_argument("--ocr", help="Path to Tesseract, used for pages with no text")
    ocr_mode.add_argument("--rapid-ocr", action="store_true", help="Use optional local ONNX OCR")
    parser.add_argument("--instrument")
    parser.add_argument("--jurisdiction", default="Unspecified")
    args = parser.parse_args()
    rows = list(extract(args.pdf, args.source_url, args.ocr, args.instrument, args.jurisdiction, args.rapid_ocr))
    if not rows:
        parser.error("No provision headings found. Check extraction or OCR.")
    args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
