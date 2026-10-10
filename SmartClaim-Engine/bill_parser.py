import os
import base64
import mimetypes
from typing import Optional, Tuple
from pypdf import PdfReader

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}


def is_image_file(file_path: str) -> bool:
    """Checks whether the file is an image format."""
    if not file_path:
        return False
    ext = os.path.splitext(file_path)[1].lower()
    return ext in IMAGE_EXTENSIONS


def load_bill_content(file_path: str) -> Tuple[Optional[str], Optional[dict]]:
    """Loads bill content from file.
    Returns a tuple of:
    - text_content: str (if PDF or text-based file)
    - image_data: dict with {"mime_type": str, "data": str (base64)} (if image file)
    """
    if not file_path or not os.path.exists(file_path):
        return None, None

    ext = os.path.splitext(file_path)[1].lower()

    # 1. Image handling (Multimodal OCR via Gemini vision)
    if ext in IMAGE_EXTENSIONS:
        try:
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
            with open(file_path, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")
            return None, {"mime_type": mime_type, "data": b64_data}
        except Exception as e:
            print(f"Error reading image bill: {e}")
            return None, None

    # 2. PDF handling
    elif ext == ".pdf":
        try:
            reader = PdfReader(file_path)
            extracted = []
            for idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    extracted.append(f"Page {idx+1}: {text.strip()}")
            return "\n\n".join(extracted).strip(), None
        except Exception as e:
            print(f"Error reading PDF bill: {e}")
            return None, None

    # 3. Plain Text / CSV / Markdown handling
    else:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read().strip(), None
        except Exception as e:
            print(f"Error reading text bill: {e}")
            return None, None
