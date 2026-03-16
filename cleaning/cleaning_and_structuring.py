import pandas as pd
import json
import re
from email import message_from_string
from email.policy import default

INPUT_FILE = "../dataset/emails.csv"
OUTPUT_FILE = "./enron_structured.jsonl"


# ----------------------------
# Helper Functions
# ----------------------------

def extract_path_metadata(file_path):
    parts = file_path.split("/")

    employee = parts[0] if len(parts) > 0 else None
    folder = parts[1] if len(parts) > 1 else None
    email_id = parts[2].replace(".", "") if len(parts) > 2 else None

    return employee, folder, email_id


def normalize_email_field(field):
    if not field:
        return []

    emails = re.split(r",|;", field)

    cleaned = []
    for e in emails:
        e = e.strip()

        match = re.search(r"<(.*?)>", e)
        if match:
            e = match.group(1)

        cleaned.append(e.lower())

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
                except:
                    pass

    else:

        try:
            body = msg.get_payload(decode=True).decode(errors="ignore")
        except:
            body = str(msg.get_payload())

    return body.strip()


def extract_attachments(msg):

    attachments = []

    for part in msg.walk():

        content_disposition = part.get("Content-Disposition")

        if content_disposition and "attachment" in content_disposition:

            payload = part.get_payload(decode=True)

            attachment = {
                "filename": part.get_filename(),
                "content_type": part.get_content_type(),
                "size": len(payload) if payload else 0
            }

            attachments.append(attachment)

    return attachments


# ----------------------------
# Main Pipeline
# ----------------------------

df = pd.read_csv(INPUT_FILE)
print(df.columns)

emails = []

for i, row in enumerate(df.itertuples(index=False)):

    file_path = row.file
    raw_message = row.message

    try:

        msg = message_from_string(raw_message, policy=default)

        employee, folder, email_id = extract_path_metadata(file_path)

        headers = {}
        for key, value in msg.items():
            headers[key] = str(value)

        body = extract_body(msg)
        attachments = extract_attachments(msg)

        email_record = {

            "file_path": file_path,
            "employee": employee,
            "folder": folder,
            "email_id": email_id,

            "message_id": msg.get("Message-ID"),
            "date": msg.get("Date"),
            "subject": msg.get("Subject"),
            "from": msg.get("From"),
            "to": normalize_email_field(msg.get("To")),
            "cc": normalize_email_field(msg.get("Cc")),
            "bcc": normalize_email_field(msg.get("Bcc")),
            "reply_to": msg.get("Reply-To"),
            "in_reply_to": msg.get("In-Reply-To"),
            "references": msg.get("References"),

            "body": body,
            "body_length": len(body),

            "attachments": attachments,
            "attachment_count": len(attachments),

            "headers": headers
        }

        emails.append(email_record)

    except Exception as e:
        print(f"Error processing row {i}: {e}")
        continue

    if i == 60:   # just for testing
        break


# write full JSON list
with open(OUTPUT_FILE, "w") as outfile:
    json.dump(emails, outfile, indent=2)
    