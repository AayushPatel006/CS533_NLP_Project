import json
import pandas as pd
import random

INPUT_FILE = "enron_structured.json"
OUTPUT_FILE = "sample_100_subject_body.xlsx"

# load dataset
with open(INPUT_FILE, "r") as f:
    data = json.load(f)

# filter records with both subject and body
filtered = []

for email in data:
    subject = email.get("subject", "")
    body = email.get("body", "")
    message_id = email.get("message_id", "")

    if subject and body:
        filtered.append({
            "message_id": message_id,
            "subject": subject.strip(),
            "body": body.strip()
        })

print("Valid emails:", len(filtered))

# shuffle dataset
random.shuffle(filtered)

# take 100 samples
sample = filtered[:100]

# convert to dataframe
df = pd.DataFrame(sample)

# save to excel
df.to_excel(OUTPUT_FILE, index=False)

print("Excel file created:", OUTPUT_FILE)