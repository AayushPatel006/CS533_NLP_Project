import pandas as pd
import re

#Accuracy: 43.43% (43/99)

# --- PATHS ---
INPUT_FILE = "/Users/chandan/Desktop/NLP/Golden Dataset - 300 rows.xlsx - Batch 1.csv"
OUTPUT_FILE = "/Users/chandan/Desktop/NLP/Batch1_90Plus_Attempt.xlsx"

def normalize(text):
    if not text or not isinstance(text, str): return ""
    return re.sub(r"\s+", " ", text.lower()).strip()

# --- HIGH-PRECISION INTENT DETECTORS ---

def is_information_override(text, subject):
    """GATEKEEPER: Features that force an 'Information' label."""
    t = normalize(text + " " + subject)
    # Specific patterns from your feature lists [cite: 1, 15]
    info_triggers = [
        "fyi", "for your information", "newsletter", "advertisement", "keynotes",
        "announcement", "broadcasting", "all employees", "distribution list",
        "thank you", "thanks", "greetings", "noted", "received", "just an update"
    ]
    # "Please be advised" is almost always Information 
    if "please be advised" in t or "please note" in t:
        return True
    return any(trigger in t for trigger in info_triggers)

def is_truly_urgent(text, subject):
    """Detects 'Hyper' users and High-Stakes business[cite: 1, 4, 15, 24]."""
    t = normalize(text + " " + subject)
    # Navneet's 'Hyper' signal: Repeated punctuation [cite: 14, 15]
    if re.search(r"\?\?+|\!\!+", t): return True
    # ASAP + Action Verb [cite: 4, 23]
    if "asap" in t or "immediately" in t or "tonight" in t:
        if any(v in t for v in ["submit", "review", "call", "send", "approve"]):
            return True
    # High-Stakes Entities from your Enron analysis [cite: 5, 20, 21, 24]
    if any(x in t for x in ["stocks", "legal", "security", "nymex", "tucson electric"]):
        return True
    return False

def is_direct_action(text, subject):
    """Detects expectation of response/task completion[cite: 10, 11, 12, 13]."""
    t, s = normalize(text), normalize(subject)
    # Subject intent: 'Question' or 'Action Required' [cite: 1, 15]
    if any(x in s for x in ["question", "action", "request"]): return True
    # Command Pattern: Verb + Noun/Object [cite: 10, 35]
    if re.search(r"(submit|review|approve|check|send|prepare) (the|this|attached|doc|report|my)", t):
        return True
    # Conversational expectations [cite: 13, 33]
    if "let me know" in t or "give me call" in t or "seeking views" in t:
        return True
    # Explicit Questions (Sentence start) [cite: 1, 26, 37]
    if t.startswith(("what", "how", "can we", "could you", "is there")):
        return True
    return False

# --- FINAL DECISION HIERARCHY ---

def assign_label_90(row):
    subj, body = str(row['subject']), str(row['body'])
    
    # 1. Urgent (High Pressure/Critical Topic) takes priority [cite: 3, 4, 5]
    if is_truly_urgent(body, subj):
        return "URGENT"
    
    # 2. Information Override (Filter out Newsletters/FYIs) [cite: 7, 8, 39]
    if is_information_override(body, subj):
        return "INFORMATION"
    
    # 3. Action Intent (Tasks/Questions) [cite: 10, 11, 14]
    if is_direct_action(body, subj):
        return "ACTION"
    
    # 4. Fallback [cite: 9]
    return "INFORMATION"

def run_90_pipeline():
    df = pd.read_csv(INPUT_FILE)
    df['Predicted Label'] = df.apply(assign_label_90, axis=1)
    
    actual = df['Final Label'].str.strip().str.upper()
    pred = df['Predicted Label'].str.strip().str.upper()
    correct = (actual == pred).sum()
    
    print(f"\n--- 90% TARGET PERFORMANCE ---")
    print(f"Accuracy: {(correct/len(df))*100:.2f}% ({correct}/{len(df)})")
    
    # Standard Excel Formatting applies here...
    df.to_excel(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    run_90_pipeline()