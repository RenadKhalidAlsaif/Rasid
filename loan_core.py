"""Core logic for the loan agent: OCR, field extraction, decisions, logs, and overrides."""
import csv
import hashlib
import json
import os
import re
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import cv2
import pandas as pd
import pytesseract

BASE_DIR = Path(__file__).resolve().parent
PAYROLL_FILE = str(BASE_DIR / "payroll.csv")
LOG_FILE = str(BASE_DIR / "decisions.jsonl")
THRESHOLDS_FILE = str(BASE_DIR / "thresholds.json")
ADJUSTMENTS_FILE = str(BASE_DIR / "adjustments.jsonl")
ESCALATIONS_FILE = str(BASE_DIR / "escalations.jsonl")
OVERRIDES_FILE = str(BASE_DIR / "overrides.jsonl")
DEMO_FILE = str(BASE_DIR / "decisions_demo.jsonl")
REGISTRY_FILE = str(BASE_DIR / "image_registry.jsonl")
REVIEWS_FILE = str(BASE_DIR / "reviews.jsonl")

MAX_AGE = timedelta(days=183)
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
REQUIRED_FIELDS = ["national_id", "full_name", "declared_income", "form_date"]
LOAN_TYPES = ["Personal", "Auto", "Home", "Business"]

# Affordability settings. The limit, tenor, and rate are assumptions to confirm.
DBR_LIMIT = 0.45           # الحد الأعلى لنسبة الالتزامات (يحتاج تأكيد من مصدر رسمي)
DEFAULT_TENOR_MONTHS = 48  # مدة القرض الافتراضية (افتراض، يحتاج تأكيد)
ANNUAL_RATE = 0.05         # معدل الفائدة السنوي (افتراض مؤقت، يحتاج تأكيد من البنك)

FIELD_AR = {
    "national_id": "رقم الهوية",
    "full_name": "الاسم",
    "declared_income": "الدخل الشهري",
    "form_date": "تاريخ الشهادة",
}

VELOCITY_LIMIT = 3
VELOCITY_WINDOW = timedelta(days=7)
REPEAT_WINDOW_DAYS = 30

FIELD_RULES = {
    "national_id": {
        "labels": ["ID / Iqama number", "National ID", "Iqama number", "ID number",
                   "Iqama", "ID", "Employee", "الموظف",
                   "رقم الهوية / الإقامة", "رقم الهوية الوطنية", "رقم الهوية", "رقم الإقامة"],
        "value": r"(\d{10})",
    },
    "full_name": {
        "labels": ["Full Name", "Applicant Name", "Customer Name", "Name", "Employee", "الموظف",
                   "الاسم الكامل", "الاسم"],
        "value": r"([^\n]+)",
    },
    "declared_income": {
        "labels": ["Total Monthly Income", "Monthly Income", "Total Income", "Income",
                   "Monthly Salary", "Total Salary", "Net Salary", "Salary",
                   "الدخل الشهري", "إجمالي الراتب", "الراتب"],
        "value": r"([\d,]+(?:\.\d+)?)",
    },
    "form_date": {
        "labels": ["Form Date", "Issue Date", "Date of Issue", "Issued On",
                   "تاريخ الإصدار", "تاريخ الاصدار", "تاريخ النموذج"],
        "value": r"(\d{4}[-/]\d{2}[-/]\d{2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})",
    },
}


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
        self.last_engine = None
        self.backend, self.reason = self._choose()

    def _unavailable_reason(self):
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
        reason = self._unavailable_reason()
        if reason is None:
            return "deepseek", "GPU and model available"
        return "tesseract", f"fallback: {reason}"

    def _load_deepseek(self):
        if self._model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
            self._model = AutoModel.from_pretrained(
                self.model_dir, trust_remote_code=True, use_safetensors=True,
                attn_implementation="eager",
            ).to(dtype=torch.bfloat16, device="cuda").eval()
        return self._tok, self._model

    def _read_tesseract(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return pytesseract.image_to_string(gray, lang="ara+eng")

    def _read_deepseek(self, image_path):
        import torch
        tok, model = self._load_deepseek()
        out_dir = tempfile.mkdtemp(prefix="deepseek_ocr_")
        with torch.inference_mode():
            model.infer(tok, prompt=self.DEEPSEEK_PROMPT, image_file=image_path,
                        output_path=out_dir, base_size=1024, image_size=640,
                        crop_mode=False, save_results=True, test_compress=True)
        # The exact output filename is not confirmed yet, so collect any text file it wrote.
        texts = [p.read_text(encoding="utf-8", errors="ignore")
                 for p in sorted(Path(out_dir).rglob("*"))
                 if p.suffix in {".md", ".mmd", ".txt"}]
        if not texts:
            raise RuntimeError(f"DeepSeek-OCR produced no text output in {out_dir}")
        text = "\n".join(texts)
        text = re.sub(r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>", "", text, flags=re.S)
        return text

    def read_text(self, image_path):
        """Return the raw OCR text for one form image."""
        if self.backend == "deepseek":
            try:
                text = self._read_deepseek(image_path)
                self.last_engine = "deepseek"
                return text
            except Exception as exc:
                if self.requested == "deepseek":
                    raise
                print(f"DeepSeek failed on {image_path} ({exc}); using Tesseract for this form.")
        self.last_engine = "tesseract"
        return self._read_tesseract(image_path)


_ocr = None


def get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = OCREngine()
    return _ocr


def load_payroll(path=PAYROLL_FILE):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {row["national_id"]: row for row in csv.DictReader(f)}


def normalize(text):
    text = text.translate(ARABIC_DIGITS)
    return re.sub(r"[ \t]+", " ", text)


def normalize_arabic(text):
    """Compare names regardless of case, hamza/ta-marbuta variants, and extra spaces."""
    text = text.strip().lower()
    for src, dst in [("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ة", "ه"), ("ى", "ي")]:
        text = text.replace(src, dst)
    return " ".join(text.split())


def find_field(text, rule):
    for label in rule["labels"]:
        pattern = r"(?<![A-Za-z])" + re.escape(label) + r"\s*[:：\-]\s*" + rule["value"]
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def clean_name(name):
    if not name:
        return None
    cleaned = re.sub(r"\s+(Monthly|Income|Total|Form|Loan)\b.*$", "", name).strip()
    return cleaned or None


def parse_date(value):
    if not value:
        return None
    value = value.replace("/", "-")
    for fmt in ("%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def is_stale(value):
    issued = parse_date(value)
    if issued is None:
        return None
    return (date.today() - issued) > MAX_AGE


def extract_fields(image_path):
    text = normalize(get_ocr().read_text(image_path))
    fields = {name: find_field(text, rule) for name, rule in FIELD_RULES.items()}
    fields["full_name"] = clean_name(fields["full_name"])
    return fields, text


def add_shadow_dbr(fields, payroll):
    """Shadow mode: record the system's obligations and the DBR. Never changes the decision."""
    record = payroll.get(fields.get("national_id"))
    if record is None or not record.get("monthly_obligations"):
        fields["system_obligations"] = None
        fields["shadow_dbr"] = None
        return
    obligations = float(record["monthly_obligations"])
    declared = fields.get("declared_income")
    declared = float(declared.replace(",", "")) if declared else 0
    fields["system_obligations"] = obligations
    fields["shadow_dbr"] = round(obligations / declared, 4) if declared else None


def monthly_installment(principal, months=DEFAULT_TENOR_MONTHS, annual_rate=ANNUAL_RATE):
    """Standard amortization payment for one month."""
    if principal <= 0:
        return 0.0
    r = annual_rate / 12
    if r == 0:
        return principal / months
    return principal * r / (1 - (1 + r) ** -months)


def affordability(fields, record):
    """Return the DBR after adding the new loan's installment, and the installment itself."""
    salary = float(record["monthly_salary"])
    obligations = float(record.get("monthly_obligations") or 0)
    installment = monthly_installment(float(fields.get("loan_amount") or 0))
    dbr = (obligations + installment) / salary
    return dbr, installment


def decide(fields, payroll):
    """Return one of: clear, reject, resubmit. Uncertain comes only from security flags."""
    missing = [k for k in REQUIRED_FIELDS if not fields.get(k)]
    if missing:
        names = "، ".join(FIELD_AR[k] for k in missing)
        return "resubmit", f"لم نتمكن من قراءة: {names}. الرجاء إعادة إرسال شهادة واضحة."

    stale = is_stale(fields["form_date"])
    if stale is None:
        return "resubmit", "تاريخ الشهادة غير واضح. الرجاء إرسال شهادة بتاريخ واضح."
    if stale:
        return "resubmit", "الشهادة أقدم من 6 أشهر. الرجاء إرسال شهادة راتب حديثة."

    nid = fields["national_id"]
    if nid not in payroll:
        return "resubmit", "رقم الهوية غير مسجل لدينا. الرجاء التأكد من الرقم وإعادة الإرسال."

    record = payroll[nid]
    form_name = normalize_arabic(fields["full_name"])
    accepted = [normalize_arabic(record["full_name"])]
    if record.get("full_name_ar"):
        accepted.append(normalize_arabic(record["full_name_ar"]))
    if not any(name and name in form_name for name in accepted):
        return "reject", f"الاسم في الشهادة لا يطابق السجل: '{fields['full_name']}'"

    declared = float(fields["declared_income"].replace(",", ""))
    actual = float(record["monthly_salary"])
    if abs(declared - actual) > 0.01:
        return "reject", f"الدخل في الشهادة ({declared:.0f}) لا يطابق السجل ({actual:.0f})"

    dbr, installment = affordability(fields, record)
    if dbr > DBR_LIMIT:
        return "reject", (
            f"نسبة الالتزامات بعد القرض {dbr:.0%} تتجاوز الحد {DBR_LIMIT:.0%} "
            f"(القسط التقديري {installment:,.0f} ريال)"
        )

    return "clear", "جميع البيانات مطابقة للسجل، ونسبة الالتزامات ضمن الحد"


def log_decision(form_name, loan_type, decision, reason, fields, duration_sec=None):
    engine = get_ocr()
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "form": form_name,
        "loan_type": loan_type,
        "decision": decision,
        "reason": reason,
        "ocr_engine": engine.last_engine or engine.backend,
        "duration_sec": duration_sec,
        "extracted": fields,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def load_decisions_df(path=LOG_FILE):
    rows = read_jsonl(path)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["month"] = df["timestamp"].dt.to_period("M").astype(str)
    df["decision"] = df["decision"].str.lower()
    return df


def load_thresholds():
    with open(THRESHOLDS_FILE, encoding="utf-8") as f:
        return json.load(f)


def save_thresholds(thresholds):
    with open(THRESHOLDS_FILE, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)


def revert_adjustment(index):
    """Undo one logged AI adjustment and record the human override."""
    adjustments = read_jsonl(ADJUSTMENTS_FILE)
    adj = adjustments[index]
    thresholds = load_thresholds()
    thresholds[adj["threshold"]] = adj["old_value"]
    save_thresholds(thresholds)
    with open(OVERRIDES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "reverted_adjustment": adj,
            "restored_value": adj["old_value"],
        }, ensure_ascii=False) + "\n")


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def register_image(sha, national_id):
    with open(REGISTRY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "sha256": sha,
            "national_id": national_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }, ensure_ascii=False) + "\n")


def check_image_reuse(sha, national_id):
    """Flag an identical image that was submitted before for a different ID."""
    for row in read_jsonl(REGISTRY_FILE):
        if row["sha256"] == sha and row["national_id"] != national_id:
            return f"Same image was submitted before for ID {row['national_id']}"
    return None


def check_data_reuse(fields):
    """Flag the same income and form date used by a different ID (ignores rows without an ID)."""
    nid = fields.get("national_id")
    income = fields.get("declared_income")
    form_date = fields.get("form_date")
    if not (nid and income and form_date):
        return None
    for row in read_jsonl(LOG_FILE):
        ex = row.get("extracted") or {}
        other = ex.get("national_id")
        if not other or other == nid:
            continue
        if ex.get("declared_income") == income and ex.get("form_date") == form_date:
            return f"Same income and form date as ID {other}"
    return None


def check_velocity(nid):
    """Flag too many applications from one ID within the window."""
    now = datetime.now()
    count = 0
    for row in read_jsonl(LOG_FILE):
        if (row.get("extracted") or {}).get("national_id") != nid:
            continue
        try:
            ts = datetime.fromisoformat(row["timestamp"])
        except (KeyError, ValueError):
            continue
        if now - ts <= VELOCITY_WINDOW:
            count += 1
    if count >= VELOCITY_LIMIT:
        return f"{count} applications from this ID in the last 7 days"
    return None


def check_repeat_after_approval(national_id, days=REPEAT_WINDOW_DAYS):
    """Flag a new request if this ID was already approved within the window."""
    if not national_id:
        return None
    cutoff = datetime.now() - timedelta(days=days)
    for row in read_jsonl(LOG_FILE):
        if row.get("decision") != "clear":
            continue
        if (row.get("extracted") or {}).get("national_id") != national_id:
            continue
        try:
            ts = datetime.fromisoformat(row["timestamp"])
        except (KeyError, ValueError):
            continue
        if ts >= cutoff:
            return (f"طلب جديد بعد موافقة سابقة خلال {days} يوماً "
                    f"(نوع القرض السابق: {row.get('loan_type')})")
    return None


def security_flags(fields, sha):
    nid = fields.get("national_id")
    if not nid:
        return []
    checks = [check_image_reuse(sha, nid), check_data_reuse(fields), check_velocity(nid)]
    return [c for c in checks if c]


def record_review(form_name, ai_decision, human_decision, note):
    with open(REVIEWS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "form": form_name,
            "ai_decision": ai_decision,
            "human_decision": human_decision,
            "note": note,
        }, ensure_ascii=False) + "\n")