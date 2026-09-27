"""PDF parsing happens in a separate process with a parent-enforced timeout."""
import io
import json
import sys

from pypdf import PdfReader


def extract(data, start, count):
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        return {"error": "Encrypted PDFs are not supported. Use the abstract."}
    total = len(reader.pages)
    if start > total:
        return {"error": f"This PDF has {total} pages. Choose a smaller start_page."}
    pages = []
    for i in range(start - 1, min(start - 1 + count, total)):
        text = (reader.pages[i].extract_text() or "").strip()
        pages.append({"page": i + 1, "text": text[:6500], "truncated": len(text) > 6500})
    if not any(p["text"] for p in pages):
        return {"error": "These pages have no extractable text. OCR is not included; use the abstract."}
    return {"total_pages": total, "pages": pages}


if __name__ == "__main__":
    try:
        result = extract(sys.stdin.buffer.read(15_000_001), int(sys.argv[1]), int(sys.argv[2]))
    except Exception:
        result = {"error": "This PDF could not be parsed. Use the abstract."}
    sys.stdout.buffer.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))
