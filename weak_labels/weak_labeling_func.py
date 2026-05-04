import json
import re
import spacy
from pathlib import Path

nlp = spacy.load("en_core_web_sm")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_JSON = PROJECT_ROOT / "cleaning" / "enron_structured_first_60_rows.jsonl"
OUTPUT_LABELED_JSON = PROJECT_ROOT / "weak_labels" / "enron_weakly_labeled.json"

def labeling_pipeline_pro(emails):
    labeled_data = []
    
    high_importance_domains = ["enron.com"] 

    for email in emails:
        body = email.get("body", "")
        subject = email.get("subject", "")
        doc = nlp(body)
        
        is_short = 1 if len(doc) < 50 else 0

        is_new_thread = 1 if "re:" not in subject.lower() else 0

        sender = email.get("from", "").lower()
        is_exec = 1 if any(dom in sender for dom in high_importance_domains) else 0

        has_request = 0
        for sent in doc.sents:
            if re.search(r"(submit|approve|review|send|please|could you)", sent.text.lower()):
                has_request = 1
                break

        has_deadline = 1 if re.search(r"(by|due|eod|tomorrow)", body.lower()) else 0

        urgency_score = has_deadline + (1 if is_exec and has_request else 0)
        
        if urgency_score >= 1:
            final_label = "Urgent"
        elif has_request or (is_short and is_new_thread):
            final_label = "Important"
        else:
            final_label = "Informational"

        email["weak_labels"] = {
            "p_actionable": has_request,
            "p_deadline_present": has_deadline,
            "is_short": is_short,
            "is_new_thread": is_new_thread,
            "final_class": final_label
        }
        labeled_data.append(email)
    
    return labeled_data

with open(INPUT_JSON, "r") as f:
    raw = f.read().strip()
    data = json.loads(raw) if raw.startswith("[") else [json.loads(line) for line in raw.splitlines() if line.strip()]

labeled_results = labeling_pipeline_pro(data[:60])

with open(OUTPUT_LABELED_JSON, "w") as outfile:
    json.dump(labeled_results, outfile, indent=2)

print(f"Successfully labeled {len(labeled_results)} emails with advanced features.")
