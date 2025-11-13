from typing import List

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None  # type: ignore


def load_pdfs(uploaded_files) -> List[str]:
    texts: List[str] = []
    for f in uploaded_files:
        if PdfReader is None:
            texts.append(f"[PDF placeholder content for {getattr(f, 'name', 'uploaded')}]")
            continue
        reader = PdfReader(f)
        content = []
        for page in reader.pages:
            try:
                content.append(page.extract_text() or "")
            except Exception:
                continue
        texts.append("\n".join(content).strip())
    return texts
