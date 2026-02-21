import io
import os
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from flask import Flask, render_template, request
from PIL import Image

from model import FraudDetector

try:
    import pytesseract
except ImportError:  # OCR remains optional if dependency is unavailable.
    pytesseract = None


FREE_EMAIL_DOMAINS = {
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "hotmail.com",
    "protonmail.com",
    "aol.com",
    "icloud.com",
    "mail.com",
}


OFFICIAL_COMPANY_DOMAINS = {
    "linkedin": {"linkedin.com"},
    "google": {"google.com"},
    "microsoft": {"microsoft.com"},
    "amazon": {"amazon.com"},
    "meta": {"meta.com", "fb.com"},
    "apple": {"apple.com"},
    "netflix": {"netflix.com"},
    "infosys": {"infosys.com"},
    "tcs": {"tcs.com"},
    "wipro": {"wipro.com"},
}

EMAIL_REGEX = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")


@dataclass
class DomainCheckResult:
    status: str
    details: str


class DomainVerifier:
    @staticmethod
    def extract_emails(text: str) -> list[str]:
        return EMAIL_REGEX.findall(text.lower())

    @staticmethod
    def infer_company_domains(text: str) -> set[str]:
        text_l = text.lower()
        inferred = set()
        for company, domains in OFFICIAL_COMPANY_DOMAINS.items():
            if company in text_l:
                inferred.update(domains)
        return inferred

    def verify(self, text: str) -> DomainCheckResult:
        emails = self.extract_emails(text)
        if not emails:
            return DomainCheckResult(
                status="No email found",
                details="No recruiter email address was detected in the content.",
            )

        inferred_domains = self.infer_company_domains(text)
        suspicious = []
        trusted = []

        for email in emails:
            domain = email.split("@")[-1]
            if domain in FREE_EMAIL_DOMAINS:
                suspicious.append(f"{email} uses a free email provider")
            elif inferred_domains and domain not in inferred_domains:
                suspicious.append(
                    f"{email} does not match inferred company domains ({', '.join(sorted(inferred_domains))})"
                )
            else:
                trusted.append(email)

        if suspicious and not trusted:
            return DomainCheckResult(
                status="Suspicious",
                details="; ".join(suspicious),
            )

        if suspicious and trusted:
            return DomainCheckResult(
                status="Mixed",
                details=f"Trusted: {', '.join(trusted)} | Suspicious: {'; '.join(suspicious)}",
            )

        return DomainCheckResult(
            status="Verified",
            details=f"All detected emails look consistent: {', '.join(trusted)}",
        )


app = Flask(__name__)
detector = FraudDetector()
verifier = DomainVerifier()


def extract_text_from_image(file_storage) -> Tuple[str, Optional[str]]:
    if pytesseract is None:
        return "", "OCR unavailable because pytesseract is not installed in this environment."

    try:
        image = Image.open(io.BytesIO(file_storage.read()))
        text = pytesseract.image_to_string(image)
        if not text.strip():
            return "", "OCR succeeded but no readable text was found in the image."
        return text, None
    except Exception as exc:  # broad so app can return a clear error in UI
        return "", f"Unable to process image for OCR: {exc}"


@app.route("/", methods=["GET", "POST"])
def index():
    result: Dict[str, str] = {}

    if request.method == "POST":
        text_input = request.form.get("job_text", "").strip()
        image_file = request.files.get("job_image")

        extracted_text = ""
        ocr_message = None

        if image_file and image_file.filename:
            extracted_text, ocr_message = extract_text_from_image(image_file)

        combined_text = "\n".join(part for part in [text_input, extracted_text] if part.strip())

        if not combined_text.strip():
            result["error"] = "Please provide job text or upload an image screenshot."
        else:
            label, confidence = detector.predict(combined_text)
            domain_result = verifier.verify(combined_text)
            result = {
                "classification": label,
                "confidence": f"{confidence * 100:.2f}%",
                "domain_status": domain_result.status,
                "domain_details": domain_result.details,
                "ocr_message": ocr_message or ("OCR extraction completed." if extracted_text else ""),
                "extracted_text": extracted_text,
            }

    return render_template("index.html", result=result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
