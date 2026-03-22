import pandas as pd
import re
import os

#Accuracy: 38.38% (38/99)

# --- PATHS ---
INPUT_FILE = "/Users/chandan/Desktop/NLP/Golden Dataset - 300 rows.xlsx - Batch 1.csv"
OUTPUT_FILE = "/Users/chandan/Desktop/NLP/Batch1_Super_Improved_Results.xlsx"

ABSTAIN = -1

def normalize(text):
    if not text or not isinstance(text, str): return ""
    return re.sub(r"\s+", " ", text.lower()).strip()

# --- NEW: ADVANCED DETECTORS ---

def has_pressure_signals(text):
    """Detects repeated punctuation or high-pressure phrases[cite: 66]."""
    if re.search(r"\?\?+|\!\!+", text): return True
    pressure_terms = ["immediately", "now", "stop", "critical", "asap", "tonight"]
    return any(term in text.lower() for term in pressure_terms)

def is_conversational_request(text):
    """Detects the expectation of a reply[cite: 56, 73]."""
    patterns = [r"let me know", r"thoughts\?", r"how about", r"can we", r"reply to"]
    return any(re.search(p, text.lower()) for p in patterns)

# --- REFINED LFs ---

def lf_urgent_signals(text, subject):
    """High-priority urgency[cite: 46, 60]."""
    t = normalize(text + " " + subject)
    # Temporal + Pressure combo
    if has_pressure_signals(t) and any(x in t for x in ["today", "deadline", "by"]):
        return 1
    # Specific high-stakes topics from your list [cite: 63, 67]
    if any(x in t for x in ["legal", "security", "stocks", "booking", "reservation"]):
        return 1
    return ABSTAIN

def lf_action_signals(text, subject):
    """Detects next steps or follow-ups[cite: 53, 57]."""
    t = normalize(text)
    s = normalize(subject)
    # Action verbs + Object pairing [cite: 70, 78]
    if re.search(r"(review|submit|approve|check|sign|send) (the|this|my|attached|doc)", t):
        return 1
    # Subject-based intent 
    if any(x in s for x in ["action", "question", "request", "please"]):
        return 1
    # Questions and conversational requests [cite: 69, 76]
    if "?" in t or is_conversational_request(t):
        return 1
    return ABSTAIN

def lf_info_signals(text):
    """Passive sharing of facts[cite: 49, 50]."""
    t = normalize(text)
    # Filter for social noise and broad announcements [cite: 77]
    if any(x in t for x in ["thanks", "thank you", "fyi", "announcement", "broadcasting"]):
        return 1
    return ABSTAIN

# --- MAIN ENGINE ---

def run_super_improved():
    if not os.path.exists(INPUT_FILE):
        print(f"File not found: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)
    results = []

    for _, row in df.iterrows():
        subj, body = str(row['subject']), str(row['body'])
        
        u = lf_urgent_signals(body, subj)
        a = lf_action_signals(body, subj)
        i = lf_info_signals(body)

        # Logic: If it triggers Urgent, it's URGENT. 
        # If it triggers Action OR (Action and Info), it's ACTION[cite: 57].
        if u == 1:
            pred = "URGENT"
        elif a == 1:
            pred = "ACTION"
        else:
            pred = "INFORMATION"
            
        results.append({
            "LF_Urgent": u, "LF_Action": a, "LF_Info": i,
            "Predicted Label": pred
        })

    analysis_df = pd.DataFrame(results)
    final_df = pd.concat([df, analysis_df], axis=1)

    # Stats
    actual = final_df['Final Label'].str.strip().str.upper()
    pred = final_df['Predicted Label'].str.strip().str.upper()
    correct = (actual == pred).sum()
    
    print(f"\n--- SUPER IMPROVED PERFORMANCE ---")
    print(f"Accuracy: {(correct/len(df))*100:.2f}% ({correct}/{len(df)})")
    
    # Save with formatting
    writer = pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter')
    final_df.to_excel(writer, index=False)
    # ... (Add green/red formatting here as in previous scripts)
    writer.close()
    print(f"Detailed analysis saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_super_improved()