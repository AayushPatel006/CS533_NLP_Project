import pandas as pd
import re
import os


#------------------------------
#ALGORITHM PERFORMANCE SUMMARY
#------------------------------
#Total Emails Processed: 100
#Correctly Predicted:    47
#Incorrectly Predicted:  53
#Accuracy Score:         47.00%
#------------------------------


# --- PATHS ---
# Using the absolute path to ensure the file is found
INPUT_FILE = "/Users/chandan/Desktop/NLP/Golden Dataset - 300 rows.xlsx - Batch 3.csv"
OUTPUT_FILE = "/Users/chandan/Desktop/NLP/Batch3_Algorithm_Analysis.xlsx"

# --- YOUR LOGIC (UNCHANGED) ---
ABSTAIN = -1

def normalize(text):
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def contains_any(text, keywords):
    return any(k in text for k in keywords)

def regex_match(text, patterns):
    return any(re.search(p, text) for p in patterns)

def has_action_and_time(text):
    action_terms = ["please", "can you", "need you", "let me know"]
    time_terms = ["today", "tomorrow", "asap", "urgent", "eod"]
    return contains_any(text, action_terms) and contains_any(text, time_terms)

# URGENCY LFs
def lf_u1_strong_urgency(text):
    text = normalize(text)
    keywords = ["asap", "urgent", "immediately", "right away", "as soon as possible"]
    return 1 if contains_any(text, keywords) else ABSTAIN

def lf_u2_deadline(text):
    text = normalize(text)
    patterns = [r"by (monday|tuesday|wednesday|thursday|friday)", r"by (today|tomorrow|tonight)", 
                r"by \d{1,2}(:\d{2})?\s?(am|pm)?", r"before \d{1,2}", r"no later than", r"deadline"]
    return 1 if regex_match(text, patterns) else ABSTAIN

def lf_u3_temporal(text):
    text = normalize(text)
    terms = ["today", "tonight", "eod", "end of day"]
    return 1 if contains_any(text, terms) else ABSTAIN

def lf_u4_action_time_combo(text):
    text = normalize(text)
    return 1 if has_action_and_time(text) else ABSTAIN

def lf_u5_scheduling(text):
    text = normalize(text)
    if "meeting" in text or "schedule" in text:
        if contains_any(text, ["today", "tomorrow", "asap"]):
            return 1
    return ABSTAIN

def lf_u6_domain(text):
    text = normalize(text)
    domains = ["security", "breach", "legal", "compliance", "medical", "emergency"]
    return 1 if contains_any(text, domains) else ABSTAIN

# ACTION LFs
def lf_a1_request(text):
    text = normalize(text)
    phrases = ["can you", "could you", "please", "i need you to", "kindly", "please confirm"]
    return 1 if contains_any(text, phrases) else ABSTAIN

def lf_a2_question(text):
    text = normalize(text)
    if "?" in text: return 1
    starters = ["what", "why", "when", "where", "who", "how", "is it", "are you", "can we", "should we"]
    return 1 if any(text.startswith(s) for s in starters) else ABSTAIN

def lf_a3_action_verbs(text):
    text = normalize(text)
    verbs = ["review", "submit", "approve", "send", "check", "confirm", "update", "complete"]
    return 1 if contains_any(text, verbs) else ABSTAIN

def lf_a4_followup(text):
    text = normalize(text)
    phrases = ["following up", "any updates", "checking in", "circling back"]
    return 1 if contains_any(text, phrases) else ABSTAIN

def lf_a5_approval(text):
    text = normalize(text)
    phrases = ["please approve", "need approval", "your approval", "let me know if this works"]
    return 1 if contains_any(text, phrases) else ABSTAIN

# INFORMATION LFs
def lf_i1_info(text):
    text = normalize(text)
    keywords = ["fyi", "for your information", "just to inform", "for reference"]
    return 1 if contains_any(text, keywords) else ABSTAIN

def lf_i2_ack(text):
    text = normalize(text)
    phrases = ["thank you", "thanks", "noted", "received", "got it"]
    return 1 if contains_any(text, phrases) else ABSTAIN

def lf_i3_broadcast(text):
    text = normalize(text)
    phrases = ["we are pleased to announce", "this is to inform", "all employees", "company-wide"]
    return 1 if contains_any(text, phrases) else ABSTAIN

def lf_i4_attachment(text):
    text = normalize(text)
    phrases = ["attached", "see attached", "find attached"]
    return 1 if contains_any(text, phrases) else ABSTAIN

# --- PROCESSING ---

LFs = {
    "LF_U1_StrongUrgency": lf_u1_strong_urgency, "LF_U2_Deadline": lf_u2_deadline, 
    "LF_U3_Temporal": lf_u3_temporal, "LF_U4_ActionTime": lf_u4_action_time_combo,
    "LF_U5_Scheduling": lf_u5_scheduling, "LF_U6_DomainCriticality": lf_u6_domain,
    "LF_A1_DirectRequest": lf_a1_request, "LF_A2_Question": lf_a2_question, 
    "LF_A3_ActionVerbs": lf_a3_action_verbs, "LF_A4_FollowUp": lf_a4_followup, "LF_A5_Approval": lf_a5_approval,
    "LF_I1_ExplicitInfo": lf_i1_info, "LF_I2_Ack": lf_i2_ack, "LF_I3_Broadcast": lf_i3_broadcast, "LF_I4_Attachment": lf_i4_attachment
}

def process_batch():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File not found at {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE) 

    def analyze_email(row):
        combined_text = f"{row['subject']} {row['body']}"
        outputs = {name: func(combined_text) for name, func in LFs.items()}
        u_votes = sum(1 for k in outputs if k.startswith("LF_U") and outputs[k] == 1)
        a_votes = sum(1 for k in outputs if k.startswith("LF_A") and outputs[k] == 1)
        
        if u_votes >= 1:
            predicted = "URGENT"
        elif a_votes >= 1:
            predicted = "ACTION"
        else:
            predicted = "INFORMATION"
            
        return pd.Series({**outputs, "Predicted Label": predicted})

    analysis_df = df.apply(analyze_email, axis=1)
    final_df = pd.concat([df, analysis_df], axis=1)

    # --- TERMINAL SUMMARY ---
    # Normalize strings for comparison (strip whitespace and uppercase)
    actual_labels = final_df['Final Label'].str.strip().str.upper()
    predicted_labels = final_df['Predicted Label'].str.strip().str.upper()
    
    correct_count = (actual_labels == predicted_labels).sum()
    incorrect_count = len(final_df) - correct_count

    print("-" * 30)
    print("ALGORITHM PERFORMANCE SUMMARY")
    print("-" * 30)
    print(f"Total Emails Processed: {len(final_df)}")
    print(f"Correctly Predicted:    {correct_count}")
    print(f"Incorrectly Predicted:  {incorrect_count}")
    print(f"Accuracy Score:         {(correct_count/len(final_df))*100:.2f}%")
    print("-" * 30)

    # --- SAVE TO EXCEL ---
    writer = pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter')
    final_df.to_excel(writer, index=False, sheet_name='Analysis')
    workbook = writer.book
    worksheet = writer.sheets['Analysis']
    green_format = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
    red_format = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
    pred_col = final_df.columns.get_loc("Predicted Label")
    final_col = final_df.columns.get_loc("Final Label")

    for row_num in range(1, len(final_df) + 1):
        actual = str(final_df.iloc[row_num-1, final_col]).strip().upper()
        predicted = str(final_df.iloc[row_num-1, pred_col]).strip().upper()
        cell_val = final_df.iloc[row_num-1, pred_col]
        if actual == predicted:
            worksheet.write(row_num, pred_col, cell_val, green_format)
        else:
            worksheet.write(row_num, pred_col, cell_val, red_format)

    writer.close()
    print(f"Excel analysis file saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_batch()

