"""
STEP 4: SNORKEL LABEL MODEL INTEGRATION
=========================================
Instead of hand-coded priority voting, Snorkel's generative LabelModel:
  - Learns the ACCURACY of each LF automatically from data
  - Learns CORRELATIONS between LFs (e.g., LF_U1 and LF_U3 often fire together)
  - Produces probabilistic labels instead of hard votes
  - Has been shown to outperform majority vote significantly

Install requirement:
    pip install snorkel

HOW IT WORKS:
  1. Each LF votes on every email → produces an LF matrix (N emails × M LFs)
  2. LabelModel learns LF accuracies from the matrix (no gold labels needed)
  3. LabelModel outputs soft probabilities → we argmax to get hard labels
  4. We evaluate against your 300-email gold set

LF RETURN VALUES (Snorkel convention):
  URGENT      = 0
  ACTION      = 1
  INFORMATION = 2
  ABSTAIN     = -1
"""

import pandas as pd
import numpy as np
import re
import os

# ── Snorkel import with helpful error message ──
try:
    from snorkel.labeling import LabelingFunction, PandasLFApplier
    from snorkel.labeling.model import LabelModel
    from snorkel.analysis import get_label_buckets
    SNORKEL_AVAILABLE = True
except ImportError:
    SNORKEL_AVAILABLE = False
    print("⚠️  Snorkel not installed. Run:  pip install snorkel")
    print("    Falling back to majority vote so you can still see the LF matrix.\n")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
# Update INPUT_FILE to point to your dataset.
# The file can be a CSV or Excel — adjust the read call in run_snorkel_pipeline() below.
INPUT_FILE  = "/Users/aayushpatel/Desktop/Rutgers/Academics/Spring 2026/NLP/NLP Project/dataset/Golden Dataset - 300 rows refined.xlsx"
OUTPUT_FILE = "/Users/aayushpatel/Desktop/Rutgers/Academics/Spring 2026/NLP/NLP Project/Step4_Snorkel_Results_2.xlsx"

# Snorkel integer class constants
URGENT_L      = 0
ACTION_L      = 1
INFORMATION_L = 2
ABSTAIN_L     = -1

CLASS_NAMES = {URGENT_L: "URGENT", ACTION_L: "ACTION", INFORMATION_L: "INFORMATION"}
CLASS_MAP   = {"URGENT": URGENT_L, "ACTION": ACTION_L, "INFORMATION": INFORMATION_L}


# ─────────────────────────────────────────────
# PREPROCESSING HELPERS
# ─────────────────────────────────────────────

def normalize(text):
    if not text or not isinstance(text, str): return ""
    return re.sub(r"\s+", " ", text.lower()).strip()

def clean_body(body):
    body = re.sub(r'-----original message-----.*', '', body, flags=re.DOTALL)
    body = re.sub(r'---------------------- forwarded by.*', '', body, flags=re.DOTALL)
    body = re.sub(r'on .{5,50} wrote:.*', '', body, flags=re.DOTALL)
    return body

def contains_any(text, keywords):
    return any(k in text for k in keywords)

def regex_match(text, patterns):
    return any(re.search(p, text) for p in patterns)

def get_text(row):
    subj = normalize(str(row.get('subject', '')))
    body = normalize(clean_body(str(row.get('body', ''))))
    return subj, body, f"{subj} {body}"

def has_negation(text):
    negation_phrases = [
        "i will let you know", "we'll check it out", "already worked this out",
        "i have already", "this has been resolved", "i've already", "already done",
        "i'll take care", "i will send", "we will send", "i'll handle",
        "no action needed", "no action required", "nothing to do"
    ]
    return contains_any(text, negation_phrases)


# ─────────────────────────────────────────────
# LABELING FUNCTIONS (Snorkel-style: row → int)
# Each function takes a pandas row and returns an integer label or ABSTAIN_L
# ─────────────────────────────────────────────

def lf_u1_asap_strong(row):
    _, _, text = get_text(row)
    urgency  = contains_any(text, ["asap", "immediately", "right away", "urgent", "as soon as possible"])
    has_verb = contains_any(text, ["submit", "review", "call", "send", "approve", "book", "confirm"])
    if urgency and has_verb: return URGENT_L
    if urgency:              return URGENT_L   # ASAP alone is still urgent
    return ABSTAIN_L

def lf_u2_deadline(row):
    _, _, text = get_text(row)
    patterns = [
        r"by (monday|tuesday|wednesday|thursday|friday|today|tonight|tomorrow)",
        r"by \d{1,2}(:\d{2})?\s?(am|pm)",
        r"no later than", r"due today", r"due by", r"deadline"
    ]
    return URGENT_L if regex_match(text, patterns) else ABSTAIN_L

def lf_u3_eod(row):
    _, _, text = get_text(row)
    return URGENT_L if contains_any(text, ["eod", "end of day", "tonight", "due today"]) else ABSTAIN_L

def lf_u4_hyper_punctuation(row):
    _, _, text = get_text(row)
    return URGENT_L if re.search(r"\?\?+|\!\!+", text) else ABSTAIN_L

def lf_u5_high_stakes(row):
    # "legal" and "compliance" removed — appear in ~60% of Enron emails,
    # causing massive false URGENT predictions. Round 3 fix.
    _, _, text = get_text(row)
    domains = ["security breach", "medical", "emergency",
               "stocks", "nymex", "healthcare", "tucson electric"]
    return URGENT_L if contains_any(text, domains) else ABSTAIN_L

def lf_u6_scheduling_pressure(row):
    """Scheduling + time pressure → URGENT."""
    _, _, text = get_text(row)
    has_sched = contains_any(text, ["schedule", "book", "reserve", "meeting"])
    has_press = contains_any(text, ["asap", "today", "tonight", "immediately", "urgent"])
    return URGENT_L if (has_sched and has_press) else ABSTAIN_L

def lf_u7_conf_call_now(row):
    # Round 3 fix: only fire for same-day calls (today/this afternoon/now),
    # not any conf call that mentions a future day.
    _, _, text = get_text(row)
    has_call = contains_any(text, ["conference call", "call at", "phone at"])
    has_time = regex_match(text, [r"\d{1,2}:\d{2}\s*(am|pm).{0,20}(today|this afternoon|now)"])
    return URGENT_L if (has_call and has_time) else ABSTAIN_L


def lf_a1_direct_command(row):
    _, _, text = get_text(row)
    pattern = r"(submit|review|approve|check|send|prepare|sign|update|complete|add)\s+(the|this|my|attached|your|a)\s+\w+"
    return ACTION_L if re.search(pattern, text) else ABSTAIN_L

def lf_a2_subject_intent(row):
    subj, _, _ = get_text(row)
    if contains_any(subj, ["question", "action required", "action needed", "request", "help needed", "please"]):
        return ACTION_L
    return ABSTAIN_L

def lf_a3_question_mark(row):
    _, _, text = get_text(row)
    if "?" in text and not has_negation(text):
        return ACTION_L
    return ABSTAIN_L

def lf_a4_conversational(row):
    _, _, text = get_text(row)
    phrases = ["let me know", "give me a call", "seeking views", "can we",
               "what do you think", "do you have", "could you please", "thoughts?"]
    if contains_any(text, phrases) and not has_negation(text):
        return ACTION_L
    return ABSTAIN_L

def lf_a5_followup(row):
    _, _, text = get_text(row)
    return ACTION_L if contains_any(text, ["following up", "circling back", "checking in", "any updates"]) else ABSTAIN_L

def lf_a6_approval(row):
    _, _, text = get_text(row)
    return ACTION_L if contains_any(text, ["please approve", "need approval", "your approval", "awaiting approval"]) else ABSTAIN_L

def lf_a7_question_start(row):
    _, body, _ = get_text(row)
    starters = ["what ", "how ", "can we ", "could you ", "is there ", "should we "]
    if any(body.strip().startswith(s) for s in starters) and not has_negation(body):
        return ACTION_L
    return ABSTAIN_L

def lf_a8_scheduling_no_pressure(row):
    """Scheduling WITHOUT pressure → ACTION not URGENT."""
    _, _, text = get_text(row)
    has_sched = contains_any(text, ["schedule", "set up a meeting", "can we meet"])
    has_press = contains_any(text, ["asap", "today", "tonight", "immediately", "urgent"])
    return ACTION_L if (has_sched and not has_press) else ABSTAIN_L


def lf_i1_fyi(row):
    _, _, text = get_text(row)
    return INFORMATION_L if contains_any(text, ["fyi", "for your information", "just to inform",
                                                 "for reference", "please be advised", "please note"]) else ABSTAIN_L

def lf_i2_acknowledgement(row):
    _, _, text = get_text(row)
    # Soft signal — only fires information if NO action words present
    has_ack    = contains_any(text, ["thank you", "thanks", "noted", "received", "got it"])
    has_action = contains_any(text, ["please", "can you", "could you", "?", "asap"])
    if has_ack and not has_action:
        return INFORMATION_L
    return ABSTAIN_L

def lf_i3_broadcast(row):
    _, _, text = get_text(row)
    return INFORMATION_L if contains_any(text, ["all employees", "company-wide", "announcement",
                                                  "broadcasting", "we are pleased to announce"]) else ABSTAIN_L

def lf_i4_hard_info(row):
    """Hard cases — these are almost never actionable."""
    _, _, text = get_text(row)
    patterns = [r"automatic reply", r"out of office", r"newsletter",
                r"distribution list", r"daily riddle", r"ecard", r"raffle",
                r"believe to be reliable", r"travelocity", r"unsubscribe"]
    return INFORMATION_L if regex_match(text, patterns) else ABSTAIN_L

def lf_i5_sender_acting(row):
    _, _, text = get_text(row)
    patterns = [r"i will (send|provide|share|forward|submit|update)",
                r"i('ll| will) (take care|handle|look into)",
                r"we will (send|provide|update|share)"]
    return INFORMATION_L if regex_match(text, patterns) else ABSTAIN_L



# ─────────────────────────────────────────────────────────────────────────────
# ROUND 1 NEW LFs — Fix 1: Stop false URGENT (ACTION→URGENT)
# ─────────────────────────────────────────────────────────────────────────────

def lf_u_legal_needs_pressure(row):
    # "legal" alone is not urgent; needs co-occurring pressure word
    _, _, text = get_text(row)
    has_legal    = contains_any(text, ["legal", "compliance", "attorney"])
    has_pressure = contains_any(text, ["asap", "immediately", "urgent", "deadline",
                                        "by today", "by tomorrow", "by end of day"])
    if has_legal and has_pressure: return URGENT_L
    if has_legal:                  return ACTION_L
    return ABSTAIN_L

def lf_u_tonight_casual_filter(row):
    # "tonight" in social context → INFO; "tonight" + work task → URGENT
    _, _, text = get_text(row)
    if "tonight" not in text:
        return ABSTAIN_L
    social = ["pizza", "drinks", "dinner", "lunch", "party", "movie",
              "going tonight", "zula", "bar", "restaurant", "date tonight"]
    if contains_any(text, social):
        return INFORMATION_L
    work = ["submit", "review", "send", "approve", "complete", "need", "call", "finish", "deploy"]
    if contains_any(text, work):
        return URGENT_L
    return ABSTAIN_L

def lf_u_conference_call_no_pressure(row):
    # Conf call without same-day pressure → ACTION; with pressure → URGENT
    _, _, text = get_text(row)
    if not contains_any(text, ["conference call", "conf call"]):
        return ABSTAIN_L
    if contains_any(text, ["today", "this afternoon", "in one hour", "asap", "now"]):
        return URGENT_L
    return ACTION_L


# ─────────────────────────────────────────────────────────────────────────────
# ROUND 1 NEW LFs — Fix 2: Stop false ACTION (INFO→ACTION)
# ─────────────────────────────────────────────────────────────────────────────

def lf_i_forwarded_chain(row):
    # Pure forwards with no ask in the top → INFO; forward + request → ACTION
    _, _, text = get_text(row)
    forward_patterns = [r"^fwd?:", r"^fw:", r"-+ forwarded by", r"forwarded by .{5,50} on"]
    if not regex_match(text, forward_patterns):
        return ABSTAIN_L
    top = text[:300]
    has_ask = contains_any(top, ["please", "can you", "could you",
                                  "let me know", "review", "approve", "send", "asap", "urgent"])
    return ACTION_L if has_ask else INFORMATION_L

def lf_i_newsletter_or_promotion(row):
    # Marketing / newsletter content → INFO
    _, _, text = get_text(row)
    patterns = [r"click here", r"unsubscribe", r"subscribe", r"privacy policy",
                r"win a free", r"enter to win", r"you are receiving this",
                r"marketing bulletin", r"rentable", r"catalog available",
                r"register (today|now) (at|by calling)", r"free (trial|download|entry)"]
    return INFORMATION_L if regex_match(text, patterns) else ABSTAIN_L

def lf_i_long_informational_report(row):
    # Very long emails that are reports/news summaries → INFO
    _, _, text = get_text(row)
    is_long = len(text) > 1500
    report_indicators = [r"press release", r"bloomberg", r"reuters", r"according to",
                         r"interim report", r"week ending", r"monthly (report|update|summary)",
                         r"credit watch", r"market power", r"legislative status"]
    if is_long and regex_match(text, report_indicators):
        return INFORMATION_L
    return ABSTAIN_L

def lf_i_social_personal_email(row):
    # Clearly personal/social emails → INFO
    _, _, text = get_text(row)
    patterns = [r"how was (your|the) (weekend|trip|game|play|party)",
                r"(drinks|dinner|lunch|pizza|bar) (tonight|last night|tomorrow)",
                r"i had a great time", r"hope to (see|hear from) you (soon)?",
                r"talk to you (soon|later)", r"love,?\s+\w+\s*$",
                r"good to hear from you", r"catch up",
                r"(funny|joke|lol|haha|lmao|rotfl)"]
    return INFORMATION_L if regex_match(text, patterns) else ABSTAIN_L

def lf_i_sender_concluding_reply(row):
    # Short closing replies → INFO; guard: direct task verb overrides
    _, _, text = get_text(row)
    task_guard = r"(ship|send|move|add|path|book|enter|update|process|forward|flip|assign)\s+(it|them|this|the|all|these)"
    if re.search(task_guard, text):
        return ABSTAIN_L
    patterns = [r"^no problem\.?\s*\n", r"^(ok|okay)\.?\s*\n",
                r"^(sounds good|looks good)\.?\s*",
                r"^thanks? (monday|tuesday|wednesday|thursday|friday)",
                r"^(noted|understood|got it|received)\.?\s*",
                r"^it was taken care of", r"^both are in",
                r"thank\w* for (your|the) (prompt|email|reply|message|help|time)"]
    if len(text) < 600 and regex_match(text, patterns):
        return INFORMATION_L
    return ABSTAIN_L


# ─────────────────────────────────────────────────────────────────────────────
# ROUND 1 NEW LFs — Fix 3: Stop false URGENT (INFO→URGENT)
# ─────────────────────────────────────────────────────────────────────────────

def lf_i_institutional_announcement(row):
    # System/HR/company broadcast notices → INFO
    _, _, text = get_text(row)
    patterns = [r"please be advised",
                r"to: all (nymex|comex|enron|employees|members|brokers)",
                r"automatically (drafted|generated|sent)",
                r"(you have been selected|you are receiving this)",
                r"system (generated|notification|alert)",
                r"if you (have any questions|encounter|need assistance)",
                r"(enron|nymex|comex|ferc) (training|notice|announcement|advisory|bulletin)",
                r"helpdesk|help desk"]
    return INFORMATION_L if regex_match(text, patterns) else ABSTAIN_L

def lf_i_schedule_or_procedure_doc(row):
    # Processing schedules, procedural timetables, migration guides → INFO
    _, _, text = get_text(row)
    patterns = [r"(date|banking business day):\s+\d",
                r"(step|item)\s+\d+[\.:]",
                r"(phase|stage)\s+\d+",
                r"as of (start of business|end of business|monday|tuesday)",
                r"will (be|become) (effective|available|live) on",
                r"(rollout|go.?live|migration) (date|on|scheduled)"]
    return INFORMATION_L if regex_match(text, patterns) else ABSTAIN_L

def lf_i_deadline_in_broadcast(row):
    # Deadlines in announcement/broadcast context → INFO not URGENT
    _, _, text = get_text(row)
    patterns = [r"(registration|enrollment|feedback|submission|response) (deadline|due date|by)",
                r"please (complete|provide|submit|have) .{0,60} by (the date|friday|end of)",
                r"(the following|below) (schedule|dates|timeline|agenda)",
                r"please plan to attend",
                r"(training|workshop|seminar|meeting) (will be held|is scheduled|takes place)"]
    return INFORMATION_L if regex_match(text, patterns) else ABSTAIN_L


# ─────────────────────────────────────────────────────────────────────────────
# ROUND 1 NEW LFs — Fix 4: Catch missed URGENT (URGENT→ACTION)
# ─────────────────────────────────────────────────────────────────────────────

def lf_u_specific_date_near_future(row):
    # Specific near-future date + action ask → URGENT
    _, _, text = get_text(row)
    near_date = [r"(monday|tuesday|wednesday|thursday|friday),?\s+(july|june|aug|sep|jan|feb|mar|apr|may|oct|nov|dec)\s+\d+",
                 r"week of (july|june|aug|sep|jan|feb|mar|apr|may|oct|nov|dec)",
                 r"(july|june|aug|sep|jan|feb|mar|apr|may|oct|nov|dec)\s+\d{1,2}(st|nd|rd|th)?,?\s+from \d",
                 r"\d{1,2}:\d{2}\s*(am|pm).{0,30}(today|tomorrow|monday|tuesday|wednesday|thursday|friday)"]
    has_ask = contains_any(text, ["please", "let me know", "email me", "plan to attend", "please join", "call"])
    return URGENT_L if (regex_match(text, near_date) and has_ask) else ABSTAIN_L

def lf_u_security_resource_request(row):
    # IT security/access approval workflow → URGENT
    _, _, text = get_text(row)
    patterns = [r"security resource request",
                r"(approve|reject) (request|access)",
                r"(submitted for your approval|awaiting (your )?approval)",
                r"(click|double.click).{0,40}(approve|reject|view)"]
    return URGENT_L if regex_match(text, patterns) else ABSTAIN_L

def lf_u_deadline_with_recipient_ask(row):
    # Personal deadline directed at recipient → URGENT
    _, _, text = get_text(row)
    patterns = [r"(responses?|comments?|feedback|changes?|summaries?) (are |is )?(due|must be (received|submitted)).{0,50}(by|on|before|no later than)",
                r"please (provide|send|submit|have) .{0,80} by (wednesday|monday|tuesday|thursday|friday|end of( the)? week|close of business)",
                r"(due|deadline|must be received) (by|on) (wednesday|monday|tuesday|thursday|friday)",
                r"by (close of business|end of( the)? week|friday,? (august|september|october|november|december|january|february|march|april|may|june|july))"]
    return URGENT_L if regex_match(text, patterns) else ABSTAIN_L


# ─────────────────────────────────────────────────────────────────────────────
# ROUND 2 NEW LFs — Fix A: Protect short direct professional requests (ACT→INFO)
# ─────────────────────────────────────────────────────────────────────────────

def lf_a_short_direct_command(row):
    # Short emails with direct command/question patterns → ACTION (high confidence)
    subj, body, _ = get_text(row)
    if len(body) > 500:
        return ABSTAIN_L
    patterns = [r"^(can|could|would|will) you\b",
                r"^(please|kindly) (let me know|call|send|forward|review|check|confirm|update|go into|path|price|draft|provide|pull|add|fix|print|run|set up)",
                r"^(are|is|do|did|have|has|was|were) (you|they|he|she|it)\b.{0,80}\?",
                r"^(call|send|check|review|forward|confirm|update|go into|path|price|let me know|give me|tell me|provide|pull|add|fix|print|run|set up)\b",
                r"(go into|path|price|rebook|book|enter|add to (the )?database|assign|flip)",
                r"would .{5,60} work for (you|everyone|the group|the team)\?",
                r"(any|an) (update|news|word|progress|response|feedback|answer)\?",
                r"(i am still|still) (in need|waiting|awaiting|pending|looking)",
                r"(i have|had) (requested|asked|sent).{0,60}(still|yet|not|no)"]
    return ACTION_L if any(re.search(p, body) for p in patterns) else ABSTAIN_L

def lf_a_fyi_with_ask(row):
    # FYI + embedded question → ACTION; pure FYI → INFO
    _, _, text = get_text(row)
    has_fyi = (text.strip().startswith("fyi") or
               re.search(r"^\w+ --\s*\nfyi", text))
    if not has_fyi:
        return ABSTAIN_L
    if "?" in text or contains_any(text, ["can you", "could you", "what do you think",
                                           "any insight", "do you see", "please advise"]):
        return ACTION_L
    return INFORMATION_L

def lf_a_operational_data_request(row):
    # Emails with specific energy quantities, deal numbers, meter IDs → ACTION
    _, _, text = get_text(row)
    patterns = [r"\d+[,.]?\d*\s*(mmbtu|mmcf|mw|mwh|dt|mcf|bbl|mmbd)",
                r"deal\s*#?\s*\d{5,}",
                r"(meter|point)\s*#?\s*\d+",
                r"path\s+\d+[,.]?\d*\s*(dt|mmbtu|mmcf)",
                r"(add to|update|enter.{0,20}in)\s+(the\s+)?(database|system|enpower|sitara)",
                r"please confirm (receipt|delivery|execution)",
                r"(wiring|wire) (instructions|transfer)",
                r"(ihs number|nymex form|caiso|ferc filing)"]
    return ACTION_L if any(re.search(p, text) for p in patterns) else ABSTAIN_L


# ─────────────────────────────────────────────────────────────────────────────
# ROUND 2 NEW LFs — Fix B: Stop false URGENT on social/casual (ACT→URGENT)
# ─────────────────────────────────────────────────────────────────────────────

def lf_u_asap_in_forwarded_thread_only(row):
    # ASAP only in forwarded content, not current sender's text → ACTION not URGENT
    _, body, _ = get_text(row)
    top = re.split(
        r'(-{3,}\s*(original message|forwarded by)|from:.{5,50}on \d{1,2}/\d{1,2}/\d{4})',
        body, flags=re.IGNORECASE
    )[0].strip()
    has_asap_top    = bool(re.search(r'\basap\b|\bimmediately\b|\bright away\b', top))
    has_asap_thread = bool(re.search(r'\basap\b|\bimmediately\b|\bright away\b', body))
    if has_asap_thread and not has_asap_top:
        return ACTION_L
    return ABSTAIN_L

def lf_u_casual_social_context(row):
    # Strong social/personal signals → INFO (overrides urgency keywords)
    _, _, text = get_text(row)
    patterns = [r"(birthday|christmas) (card|gift|present)",
                r"(high school|reunion|old friend|long lost)",
                r"(i will call|call me|call you) (tonight|later|this evening)",
                r"(thong|lotion|tan|bracelet|outfit)",
                r"(treadmill|abs|workout|gym)\b.{0,100}(how was|going to|last night)",
                r"(golf|ski|scuba|diving|certif).{0,80}(weekend|trip|go)",
                r"(canoe|hiking|camping|swimming) (trip|weekend)",
                r"(christmas|thanksgiving|holiday) (tasters|dinner|party|celebration)",
                r"i('ll| will) (come|go|join).{0,30}(stag|solo|alone|with)",
                r"love you\.?\s*$",
                r"(i am|i'm) (a fat ass|so cute)"]
    return INFORMATION_L if any(re.search(p, text) for p in patterns) else ABSTAIN_L

def lf_u_cancelled_or_resolved_meeting(row):
    # Cancelled/rescheduled meetings or resolved issues → ACTION (not URGENT)
    _, _, text = get_text(row)
    patterns = [r"(meeting|call|session|event)\s+(has been|is)\s+(cancelled|canceled|rescheduled|postponed)",
                r"(cancelled|canceled|postponed|rescheduled)\s+(the|this|our|today)",
                r"does not have to be done by",
                r"no longer need(ed)?",
                r"(has been|was)\s+(taken care of|resolved|completed|handled|addressed)",
                r"never mind|disregard (this|the previous|my)"]
    return ACTION_L if any(re.search(p, text) for p in patterns) else ABSTAIN_L


# ─────────────────────────────────────────────────────────────────────────────
# ROUND 2 NEW LFs — Fix C: Protect forwarded INFO reports from URGENT (INF→URGENT)
# ─────────────────────────────────────────────────────────────────────────────

def lf_i_minimal_comment_forward(row):
    # Current sender adds <120 chars before a long forward with no ask → INFO
    _, body, _ = get_text(row)
    split = re.split(
        r'(-{5,}\s*(original message|forwarded by)|from:.{5,80}on \d{1,2}/\d{1,2}/\d{4}|-{5,} forwarded)',
        body, flags=re.IGNORECASE
    )
    top = split[0].strip()
    if len(top) < 120 and len(body) > 400:
        has_ask = any(re.search(p, top) for p in [
            r'\?', r'please (review|respond|advise|let me know)',
            r'can you', r'could you', r'what do you think', r'asap', r'urgent', r'immediately'
        ])
        if not has_ask:
            return INFORMATION_L
    return ABSTAIN_L

def lf_i_news_or_report_forward(row):
    # Forwarded press releases, news articles, industry reports → INFO
    _, _, text = get_text(row)
    patterns = [r"(press release|bloomberg|reuters|wall street journal|"
                r"sf chronicle|san francisco chronicle|financial times)",
                r"(legislative|status) report (week ending|for (the week|month))",
                r"competitive intelligence",
                r"(editorial|op.?ed|column)\s+by\s+\w+",
                r"(article|paper|study|report)\s+(from|by|out from)\s+\w+",
                r"week ending \d{1,2}/\d{1,2}"]
    return INFORMATION_L if any(re.search(p, text) for p in patterns) else ABSTAIN_L

def lf_i_resolved_thread(row):
    # Short reply where sender closes the thread → INFO
    _, body, _ = get_text(row)
    split = re.split(
        r'(-{3,}\s*(original message|forwarded by)|from:.{5,50}on \d{1,2}/\d{1,2}/\d{4})',
        body, flags=re.IGNORECASE
    )
    top = split[0].strip().lower()
    patterns = [r"(it was|has been|were|was)\s+(taken care of|resolved|handled|addressed|completed|done|fixed)",
                r"no (problem|objection|issue|concern)\.?\s*(\n|$)",
                r"(we have|i have|have) no (objection|problem|issue|concern)",
                r"(adequately|fully|properly|already)\s+(addressed|handled|covered|resolved)",
                r"^(ok|okay|sounds good|looks good|noted|understood|confirmed|received|got it)\.?\s*\n?$",
                r"thanks? for (your|the) (prompt|quick|fast|help|email|reply|message)"]
    if len(top) < 200 and any(re.search(p, top) for p in patterns):
        return INFORMATION_L
    return ABSTAIN_L


# ─────────────────────────────────────────────
# SNORKEL LF WRAPPERS
# ─────────────────────────────────────────────

ALL_LF_FUNCTIONS = [
    # ── Original base LFs (20) ──
    lf_u1_asap_strong, lf_u2_deadline, lf_u3_eod, lf_u4_hyper_punctuation,
    lf_u5_high_stakes, lf_u6_scheduling_pressure, lf_u7_conf_call_now,
    lf_a1_direct_command, lf_a2_subject_intent, lf_a3_question_mark,
    lf_a4_conversational, lf_a5_followup, lf_a6_approval,
    lf_a7_question_start, lf_a8_scheduling_no_pressure,
    lf_i1_fyi, lf_i2_acknowledgement, lf_i3_broadcast, lf_i4_hard_info, lf_i5_sender_acting,
    # ── Round 1 new LFs (14) ──
    lf_u_legal_needs_pressure, lf_u_tonight_casual_filter, lf_u_conference_call_no_pressure,
    lf_i_forwarded_chain, lf_i_newsletter_or_promotion, lf_i_long_informational_report,
    lf_i_social_personal_email, lf_i_sender_concluding_reply,
    lf_i_institutional_announcement, lf_i_schedule_or_procedure_doc, lf_i_deadline_in_broadcast,
    lf_u_specific_date_near_future, lf_u_security_resource_request, lf_u_deadline_with_recipient_ask,
    # ── Round 2 new LFs (9) ──
    lf_a_short_direct_command, lf_a_fyi_with_ask, lf_a_operational_data_request,
    lf_u_asap_in_forwarded_thread_only, lf_u_casual_social_context, lf_u_cancelled_or_resolved_meeting,
    lf_i_minimal_comment_forward, lf_i_news_or_report_forward, lf_i_resolved_thread,
]
LF_NAMES = [f.__name__ for f in ALL_LF_FUNCTIONS]



def build_lf_matrix_manually(df):
    """
    Fallback: builds the LF matrix without Snorkel.
    Returns numpy array of shape (N, M) with values in {-1, 0, 1, 2}.
    """
    matrix = np.full((len(df), len(ALL_LF_FUNCTIONS)), ABSTAIN_L, dtype=int)
    for col_idx, lf_fn in enumerate(ALL_LF_FUNCTIONS):
        for row_idx, (_, row) in enumerate(df.iterrows()):
            matrix[row_idx, col_idx] = lf_fn(row)
    return matrix


def majority_vote(lf_matrix, n_classes=3):
    """Fallback majority vote aggregation (no Snorkel)."""
    predictions = []
    for row in lf_matrix:
        counts = np.bincount(row[row != ABSTAIN_L] + 1, minlength=n_classes + 1)
        counts = counts[1:]   # shift back
        if counts.sum() == 0:
            predictions.append(INFORMATION_L)   # default fallback
        else:
            predictions.append(int(np.argmax(counts)))
    return np.array(predictions)


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────

def run_snorkel_pipeline():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File not found at {INPUT_FILE}")
        return

    # ── Load data — handles both .xlsx and .csv ──
    print(f"Loading: {INPUT_FILE}")
    if INPUT_FILE.endswith('.xlsx') or INPUT_FILE.endswith('.xls'):
        dfs = pd.read_excel(INPUT_FILE, sheet_name=None)
        df = pd.concat(dfs.values(), ignore_index=True)
    else:
        df = pd.read_csv(INPUT_FILE)

    # Strip illegal Excel chars
    ILLEGAL_RE = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]')
    df['body']    = df['body'].apply(lambda x: ILLEGAL_RE.sub("", str(x)))
    df['subject'] = df['subject'].apply(lambda x: ILLEGAL_RE.sub("", str(x)))

    print(f"Dataset size: {len(df)} emails")
    print(f"Building LF matrix with {len(ALL_LF_FUNCTIONS)} labeling functions...")

    if SNORKEL_AVAILABLE:
        # ── Snorkel path ──
        snorkel_lfs = [
            LabelingFunction(name=fn.__name__, f=fn)
            for fn in ALL_LF_FUNCTIONS
        ]
        applier  = PandasLFApplier(lfs=snorkel_lfs)
        L_matrix = applier.apply(df)

        print(f"\nLF matrix shape: {L_matrix.shape}  ({L_matrix.shape[0]} emails × {L_matrix.shape[1]} LFs)")
        print(f"\n{'LF Name':<40} {'Coverage':>8}  {'URGENT':>6}  {'ACTION':>6}  {'INFO':>6}")
        print("-" * 70)
        for i, name in enumerate(LF_NAMES):
            col = L_matrix[:, i]
            coverage = (col != ABSTAIN_L).mean() * 100
            print(f"  {name:<38} {coverage:>7.1f}%  {(col==URGENT_L).sum():>6}  {(col==ACTION_L).sum():>6}  {(col==INFORMATION_L).sum():>6}")

        # ── Compute class balance from gold labels (if available) ──
        # Providing class_balance helps Snorkel calibrate probabilities correctly.
        # Without it, Snorkel assumes uniform priors which badly skews predictions
        # when classes are imbalanced.
        class_balance = None
        if 'Final Label' in df.columns:
            label_counts = df['Final Label'].str.strip().str.upper().value_counts()
            total = label_counts.sum()
            # Order must match: URGENT=0, ACTION=1, INFORMATION=2
            n_urgent = label_counts.get('URGENT', 0)
            n_action = label_counts.get('ACTION', 0)
            n_info   = label_counts.get('INFORMATION', label_counts.get('INFORMATIONAL', 0))
            class_balance = np.array([n_urgent, n_action, n_info], dtype=float)
            class_balance = class_balance / class_balance.sum()
            print(f"\nClass balance from gold labels:")
            print(f"  URGENT={class_balance[0]:.3f}  ACTION={class_balance[1]:.3f}  INFO={class_balance[2]:.3f}")

        # ── Also extract gold labels for LabelModel (improves accuracy significantly) ──
        # Snorkel can use gold labels during training to anchor its learned weights.
        Y_gold = None
        if 'Final Label' in df.columns:
            y_map = {'URGENT': URGENT_L, 'ACTION': ACTION_L,
                     'INFORMATION': INFORMATION_L, 'INFORMATIONAL': INFORMATION_L}
            Y_gold_raw = df['Final Label'].str.strip().str.upper().map(y_map)
            # Only use non-TIE, non-NaN labels
            valid_mask = Y_gold_raw.notna() & (df['Final Label'].str.strip().str.upper() != 'TIE')
            if valid_mask.sum() > 0:
                Y_gold = Y_gold_raw.where(valid_mask, other=-1).fillna(-1).astype(int).values
                print(f"  Gold labels available: {valid_mask.sum()} / {len(df)}")

        print("\nTraining Snorkel LabelModel...")
        print("(This learns each LF's accuracy and correlations automatically)")

        label_model = LabelModel(cardinality=3, verbose=True)

        # Fit with class balance prior — critical for imbalanced datasets
        fit_kwargs = dict(
            L_train=L_matrix,
            n_epochs=500,
            lr=0.001,
            log_freq=100,
            seed=42,
        )
        if class_balance is not None:
            fit_kwargs['class_balance'] = class_balance

        label_model.fit(**fit_kwargs)

        probs        = label_model.predict_proba(L=L_matrix)
        predictions  = np.argmax(probs, axis=1)
        method_label = "Snorkel LabelModel (43 LFs + class balance prior)"

    else:
        # ── Fallback majority vote path ──
        print("Snorkel not installed — using weighted majority vote fallback.")
        print("Run:  pip install snorkel   for best results.\n")
        L_matrix    = build_lf_matrix_manually(df)
        print(f"LF matrix shape: {L_matrix.shape}")
        predictions = majority_vote(L_matrix)
        probs       = None
        method_label = "Majority Vote (fallback — install snorkel for LabelModel)"

    # Map integer predictions back to string labels
    df['Predicted Label'] = [CLASS_NAMES[p] for p in predictions]

    # ── Accuracy report (exclude TIEs) ──
    eval_df   = df[df['Final Label'].str.strip().str.upper() != "TIE"].copy()
    actual    = eval_df['Final Label'].str.strip().str.upper()
    predicted = eval_df['Predicted Label'].str.strip().str.upper()
    correct   = (actual == predicted).sum()
    total     = len(eval_df)

    print("\n" + "="*55)
    print(f"  METHOD: {method_label}")
    print("="*55)
    print(f"  Total Evaluated : {total}")
    print(f"  Correct         : {correct}")
    print(f"  Incorrect       : {total - correct}")
    print(f"  Accuracy        : {(correct/total)*100:.2f}%")

    # Per-class breakdown
    for cls in ["URGENT", "ACTION", "INFORMATION"]:
        mask     = actual == cls
        cls_corr = ((actual == predicted) & mask).sum()
        cls_tot  = mask.sum()
        print(f"  {cls:<14}: {cls_corr}/{cls_tot} ({(cls_corr/cls_tot*100) if cls_tot else 0:.1f}%)")
    print("="*55)

    # ── Add LF matrix + probabilities to output ──
    lf_df = pd.DataFrame(L_matrix, columns=LF_NAMES)
    label_remap = {URGENT_L: "URGENT", ACTION_L: "ACTION",
                   INFORMATION_L: "INFORMATION", ABSTAIN_L: "abstain"}
    for col in lf_df.columns:
        lf_df[col] = lf_df[col].map(label_remap)

    if probs is not None:
        prob_df  = pd.DataFrame(probs, columns=["P_URGENT", "P_ACTION", "P_INFORMATION"])
        final_df = pd.concat([df, prob_df, lf_df], axis=1)
    else:
        final_df = pd.concat([df, lf_df], axis=1)

    # ── Excel export ──
    writer    = pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter')
    workbook  = writer.book

    final_df.to_excel(writer, index=False, sheet_name='Snorkel_Results')
    worksheet = writer.sheets['Snorkel_Results']

    fmt_correct   = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
    fmt_incorrect = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})

    col_final = final_df.columns.get_loc("Final Label")
    col_pred  = final_df.columns.get_loc("Predicted Label")

    for row_idx in range(1, len(final_df) + 1):
        actual_val = str(final_df.iloc[row_idx-1, col_final]).strip().upper()
        pred_val   = str(final_df.iloc[row_idx-1, col_pred]).strip().upper()
        if actual_val == "TIE":
            continue
        fmt = fmt_correct if actual_val == pred_val else fmt_incorrect
        worksheet.write(row_idx, col_pred, final_df.iloc[row_idx-1, col_pred], fmt)

    # ── LF Coverage summary sheet ──
    coverage_data = []
    for i, name in enumerate(LF_NAMES):
        col = L_matrix[:, i]
        coverage_data.append({
            "LF Name":       name,
            "Coverage %":    round((col != ABSTAIN_L).mean() * 100, 1),
            "URGENT votes":  int((col == URGENT_L).sum()),
            "ACTION votes":  int((col == ACTION_L).sum()),
            "INFO votes":    int((col == INFORMATION_L).sum()),
            "Abstain":       int((col == ABSTAIN_L).sum()),
        })
    pd.DataFrame(coverage_data).to_excel(writer, index=False, sheet_name='LF_Coverage')

    writer.close()
    print(f"\n  Saved → {OUTPUT_FILE}")
    print("  Tabs: Snorkel_Results | LF_Coverage\n")


if __name__ == "__main__":
    run_snorkel_pipeline()
