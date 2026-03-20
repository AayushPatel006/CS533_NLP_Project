import re

ABSTAIN = -1

def normalize(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def contains_any(text, keywords):
    return any(k in text for k in keywords)

def regex_match(text, patterns):
    return any(re.search(p, text) for p in patterns)

# Action + Time Combination Detector

def has_action_and_time(text):
    action_terms = ["please", "can you", "need you", "let me know"]
    time_terms = ["today", "tomorrow", "asap", "urgent", "eod"]
    
    return contains_any(text, action_terms) and contains_any(text, time_terms)

# URGENCY LFs
# LF_U1: Strong Urgency (High Precision)
def lf_u1_strong_urgency(text):
    text = normalize(text)
    
    keywords = [
        "asap", "urgent", "immediately",
        "right away", "as soon as possible"
    ]
    
    return 1 if contains_any(text, keywords) else ABSTAIN

# LF_U2: Deadline Detection (Robust)
def lf_u2_deadline(text):
    text = normalize(text)
    
    patterns = [
        r"by (monday|tuesday|wednesday|thursday|friday)",
        r"by (today|tomorrow|tonight)",
        r"by \d{1,2}(:\d{2})?\s?(am|pm)?",
        r"before \d{1,2}",
        r"no later than",
        r"deadline"
    ]
    
    return 1 if regex_match(text, patterns) else ABSTAIN

# LF_U3: Temporal Pressure (Contextual)
def lf_u3_temporal(text):
    text = normalize(text)
    
    terms = ["today", "tonight", "eod", "end of day"]
    
    return 1 if contains_any(text, terms) else ABSTAIN

# LF_U4: Urgent + Action (Composite LF)
def lf_u4_action_time_combo(text):
    text = normalize(text)
    
    return 1 if has_action_and_time(text) else ABSTAIN

# LF_U5: Scheduling Urgency
def lf_u5_scheduling(text):
    text = normalize(text)
    
    if "meeting" in text or "schedule" in text:
        if contains_any(text, ["today", "tomorrow", "asap"]):
            return 1
    
    return ABSTAIN

# LF_U6: Domain Criticality
def lf_u6_domain(text):
    text = normalize(text)
    
    domains = [
        "security", "breach", "legal",
        "compliance", "medical", "emergency"
    ]
    
    return 1 if contains_any(text, domains) else ABSTAIN

# 3. ACTION LFs (Advanced)
# LF_A1: Direct Request
def lf_a1_request(text):
    text = normalize(text)
    
    phrases = [
        "can you", "could you", "please",
        "i need you to", "kindly",
        "please confirm"
    ]
    
    return 1 if contains_any(text, phrases) else ABSTAIN

# LF_A2: Question Detection (Improved)
def lf_a2_question(text):
    text = normalize(text)
    
    if "?" in text:
        return 1
    
    starters = [
        "what", "why", "when", "where", "who", "how",
        "is it", "are you", "can we", "should we"
    ]
    
    return 1 if any(text.startswith(s) for s in starters) else ABSTAIN

# LF_A3: Action Verbs (Contextual)
def lf_a3_action_verbs(text):
    text = normalize(text)
    
    verbs = [
        "review", "submit", "approve",
        "send", "check", "confirm",
        "update", "complete"
    ]
    
    return 1 if contains_any(text, verbs) else ABSTAIN

# LF_A4: Follow-up Signals
def lf_a4_followup(text):
    text = normalize(text)
    
    phrases = [
        "following up", "any updates",
        "checking in", "circling back"
    ]
    
    return 1 if contains_any(text, phrases) else ABSTAIN

# LF_A5: Approval Seeking
def lf_a5_approval(text):
    text = normalize(text)
    
    phrases = [
        "please approve",
        "need approval",
        "your approval",
        "let me know if this works"
    ]
    
    return 1 if contains_any(text, phrases) else ABSTAIN

# 4. INFORMATION LFs (Advanced)
# LF_I1: Explicit Info
def lf_i1_info(text):
    text = normalize(text)
    
    keywords = [
        "fyi", "for your information",
        "just to inform", "for reference"
    ]
    
    return 1 if contains_any(text, keywords) else ABSTAIN

# LF_I2: Acknowledgment
def lf_i2_ack(text):
    text = normalize(text)
    
    phrases = [
        "thank you", "thanks",
        "noted", "received", "got it"
    ]
    
    return 1 if contains_any(text, phrases) else ABSTAIN

# LF_I3: Broadcast Detection
def lf_i3_broadcast(text):
    text = normalize(text)
    
    phrases = [
        "we are pleased to announce",
        "this is to inform",
        "all employees",
        "company-wide"
    ]
    
    return 1 if contains_any(text, phrases) else ABSTAIN

# LF_I4: Attachment Only (Weak but Useful)
def lf_i4_attachment(text):
    text = normalize(text)
    
    phrases = [
        "attached", "see attached",
        "find attached"
    ]
    
    return 1 if contains_any(text, phrases) else ABSTAIN



# 5. Applying All LFs
LFs = [
    lf_u1_strong_urgency,
    lf_u2_deadline,
    lf_u3_temporal,
    lf_u4_action_time_combo,
    lf_u5_scheduling,
    lf_u6_domain,
    
    lf_a1_request,
    lf_a2_question,
    lf_a3_action_verbs,
    lf_a4_followup,
    lf_a5_approval,
    
    lf_i1_info,
    lf_i2_ack,
    lf_i3_broadcast,
    lf_i4_attachment
]

# Apply to Dataset

def apply_lfs(text):
    results = [lf(text) for lf in LFs]
    return results

# Convert LF Outputs → Final Label
def assign_label(text):
    text = normalize(text)
    
    urgency_votes = sum([
        lf_u1_strong_urgency(text) == 1,
        lf_u2_deadline(text) == 1,
        lf_u3_temporal(text) == 1,
        lf_u4_action_time_combo(text) == 1,
        lf_u5_scheduling(text) == 1,
        lf_u6_domain(text) == 1
    ])
    
    action_votes = sum([
        lf_a1_request(text) == 1,
        lf_a2_question(text) == 1,
        lf_a3_action_verbs(text) == 1,
        lf_a4_followup(text) == 1,
        lf_a5_approval(text) == 1
    ])
    
    info_votes = sum([
        lf_i1_info(text) == 1,
        lf_i2_ack(text) == 1,
        lf_i3_broadcast(text) == 1,
        lf_i4_attachment(text) == 1
    ])
    
    if urgency_votes >= 1:
        return "URGENT"
    elif action_votes >= 1:
        return "ACTION"
    else:
        return "INFORMATION"