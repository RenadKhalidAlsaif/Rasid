import os
import tempfile
from pathlib import Path

import cv2
import pytesseract


class OCREngine:
    TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    DEFAULT_MODEL_DIR = r"C:\Users\PCD\OneDrive\Apps\loan-agent"
    DEEPSEEK_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."

    def __init__(self, backend=None, model_dir=None):
        self.requested = backend or os.environ.get("LOAN_OCR_BACKEND", "auto")
        self.model_dir = model_dir or os.environ.get("LOAN_DEEPSEEK_DIR", self.DEFAULT_MODEL_DIR)
        pytesseract.pytesseract.tesseract_cmd = self.TESSERACT_CMD
        self._tok = None
        self._model = None
        self.backend, self.reason = self._choose()
        print(f"OCR engine: {self.backend} ({self.reason})")

    def _unavailable_reason(self):
        """Return None if DeepSeek can run here, otherwise a short reason."""
        if not os.path.isdir(self.model_dir) and "/" not in self.model_dir:
            return f"model folder not found: {self.model_dir}"
        try:
            import torch
        except ImportError:
            return "torch is not installed"
        if not torch.cuda.is_available():
            return "no CUDA GPU detected"
        try:
            import transformers  # noqa: F401
        except ImportError:
            return "transformers is not installed"
        return None

    def _choose(self):
        if self.requested == "tesseract":
            return "tesseract", "set to tesseract"
        if self.requested == "deepseek":
            reason = self._unavailable_reason()
            if reason:
                raise RuntimeError(f"DeepSeek requested but unavailable: {reason}")
            return "deepseek", "set to deepseek"
        if self.requested == "auto":
            reason = self._unavailable_reason()
            if reason is None:
                return "deepseek", "GPU and model available"
            return "tesseract", f"fallback: {reason}"
        raise ValueError(f"Unknown backend: {self.requested!r}")

    def _load_deepseek(self):
        if self._model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
            self._model = AutoModel.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
                use_safetensors=True,
                attn_implementation="eager",
            ).to(dtype=torch.bfloat16, device="cuda").eval()
        return self._tok, self._model

    def _read_tesseract(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return pytesseract.image_to_string(gray)

    def _read_deepseek(self, image_path):
        import torch

        tok, model = self._load_deepseek()
        out_dir = tempfile.mkdtemp(prefix="deepseek_ocr_")
        with torch.inference_mode():
            model.infer(
                tok,
                prompt=self.DEEPSEEK_PROMPT,
                image_file=image_path,
                output_path=out_dir,
                base_size=1024,
                image_size=640,
                crop_mode=False,
                save_results=True,
                test_compress=True,
            )
        # The exact output filename is not confirmed yet, so collect any text file it wrote.
        texts = [
            p.read_text(encoding="utf-8", errors="ignore")
            for p in sorted(Path(out_dir).rglob("*"))
            if p.suffix in {".md", ".mmd", ".txt"}
        ]
        if not texts:
            raise RuntimeError(f"DeepSeek-OCR produced no text output in {out_dir}")
        return "\n".join(texts)

    def read_text(self, image_path):
        """Return the raw OCR text for one form image."""
        if self.backend == "deepseek":
            try:
                return self._read_deepseek(image_path)
            except Exception as exc:
                if self.requested == "deepseek":
                    raise
                print(f"DeepSeek failed on {image_path} ({exc}); using Tesseract for this form.")
        return self._read_tesseract(image_path)


ocr = OCREngine()   # create once, reuse for every form