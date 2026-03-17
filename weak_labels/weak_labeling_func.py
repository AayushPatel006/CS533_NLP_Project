import json
import re
import spacy

# Load spaCy for better tokenization and segmentation [cite: 51]
nlp = spacy.load("en_core_web_sm")

INPUT_JSON = "/Users/chandan/Desktop/NLP/enron_structured.json"
OUTPUT_LABELED_JSON = "/Users/chandan/Desktop/NLP/enron_weakly_labeled.json"

def labeling_pipeline_pro(emails):
    labeled_data = []
    
    # Define "high-importance" senders for LF_sender_role [cite: 70]
    high_importance_domains = ["enron.com"] 
    # In a real scenario, you'd use a list of executive names/IDs

    for email in emails:
        body = email.get("body", "")
        subject = email.get("subject", "")
        doc = nlp(body) # Tokenization/Segmentation [cite: 51]
        
        # 1. LF_email_length: Short emails often indicate quick actions 
        is_short = 1 if len(doc) < 50 else 0

        # 2. LF_thread_position: Initial messages more likely actionable [cite: 71]
        # Assuming 'folder' or metadata indicates if it's a new thread
        is_new_thread = 1 if "re:" not in subject.lower() else 0

        # 3. LF_sender_role: Higher weight for specific senders [cite: 70]
        sender = email.get("from", "").lower()
        is_exec = 1 if any(dom in sender for dom in high_importance_domains) else 0

        # 4. Refined Linguistic Cues using Tokenization [cite: 64, 65]
        # We look for specific modal verbs or request patterns in sentences
        has_request = 0
        for sent in doc.sents:
            if re.search(r"(submit|approve|review|send|please|could you)", sent.text.lower()):
                has_request = 1
                break

        # 5. Temporal Cues [cite: 62]
        has_deadline = 1 if re.search(r"(by|due|eod|tomorrow)", body.lower()) else 0

        # Final Mapping Logic [cite: 86]
        # Incorporate Metadata for "Personalization" [cite: 20, 100]
        urgency_score = has_deadline + (1 if is_exec and has_request else 0)
        
        if urgency_score >= 1:
            final_label = "Urgent" # [cite: 87, 88]
        elif has_request or (is_short and is_new_thread):
            final_label = "Important" # [cite: 89]
        else:
            final_label = "Informational" # [cite: 90]

        email["weak_labels"] = {
            "p_actionable": has_request,
            "p_deadline_present": has_deadline,
            "is_short": is_short,
            "is_new_thread": is_new_thread,
            "final_class": final_label
        }
        labeled_data.append(email)
    
    return labeled_data

# Execute
with open(INPUT_JSON, "r") as f:
    data = json.load(f)

labeled_results = labeling_pipeline_pro(data[:60])

with open(OUTPUT_LABELED_JSON, "w") as outfile:
    json.dump(labeled_results, outfile, indent=2)

print(f"Successfully labeled {len(labeled_results)} emails with advanced features.")