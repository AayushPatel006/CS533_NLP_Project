import re
from email import message_from_string
from email.policy import default
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "dataset" / "emails_10k_sample.csv"
OUTPUT_FILE = PROJECT_ROOT / "dataset" / "emails_10k_structured.csv"


def normalize_email_field(field):
    if not field:
        return []
    emails = re.split(r",|;", str(field))
    cleaned = []
    for email in emails:
        email = email.strip()
        match = re.search(r"<(.*?)>", email)
        if match:
            email = match.group(1)
        if email:
            cleaned.append(email.lower())
    return cleaned


def extract_body(msg):
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition"))
            if content_type == "text/plain" and "attachment" not in disposition:
                try:
                    body += part.get_payload(decode=True).decode(errors="ignore")
                except Exception:
                    pass
    else:
        try:
            payload = msg.get_payload(decode=True)
            if payload is None:
                body = str(msg.get_payload())
            else:
                body = payload.decode(errors="ignore")
        except Exception:
            body = str(msg.get_payload())
    return body.strip()


def build_structured_dataset():
    df = pd.read_csv(INPUT_FILE)
    rows = []

    for row in df.itertuples(index=False):
        try:
            msg = message_from_string(str(row.message), policy=default)
            rows.append(
                {
                    "file": str(row.file),
                    "message_id": str(msg.get("Message-ID") or ""),
                    "date": str(msg.get("Date") or ""),
                    "from": str(msg.get("From") or ""),
                    "to": normalize_email_field(msg.get("To")),
                    "cc": normalize_email_field(msg.get("Cc")),
                    "bcc": normalize_email_field(msg.get("Bcc")),
                    "subject": str(msg.get("Subject") or ""),
                    "body": extract_body(msg),
                }
            )
        except Exception:
            continue

    out = pd.DataFrame(rows)
    out = out[(out["subject"].astype(str).str.strip() != "") | (out["body"].astype(str).str.strip() != "")]
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)

    print(f"Structured rows saved: {len(out)}")
    print(f"Output → {OUTPUT_FILE}")


if __name__ == "__main__":
    build_structured_dataset()
