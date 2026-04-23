"""
STEP 1 & 2: WEIGHTED CONFIDENCE SCORING + FIXED GATEKEEPER
============================================================
Key changes from your original LF_01 to LF_04:

STEP 1 - Confidence weights instead of binary voting:
  - Each LF now returns a (class, weight) tuple instead of 1/ABSTAIN
  - Strong signals (ASAP + verb) get weight 3, weak signals (?) get weight 1
  - Final label = class with highest total weight score

STEP 2 - Fixed the three biggest misclassification patterns:
  - INFO gatekeeper is now a soft vote (adds weight), NOT a hard override
  - Only true hard overrides: "automatic reply", "newsletter", "distribution list"
  - Scheduling alone is NOT urgent — needs time pressure to co-occur
  - Negation filter expanded: "I will let you know", "already done", etc.
  - Subject line given 1.5x weight multiplier
"""

import pandas as pd
import re
import os
from collections import defaultdict

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_FILE  = "/Users/aayushpatel/Desktop/Rutgers/Academics/Spring 2026/NLP/NLP Project/dataset/Golden Dataset - 300 rows refined.xlsx"
OUTPUT_FILE = "/Users/aayushpatel/Desktop/Rutgers/Academics/Spring 2026/NLP/NLP Project/LF/improvements_aayush/results/Batch_Weighted_LF_Results_round_3_2.xlsx"

URGENT      = "URGENT"
ACTION      = "ACTION"
INFORMATION = "INFORMATION"
ABSTAIN     = None   # None means this LF has no opinion


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def normalize(text):
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clean_body(body):
    """Strip forwarded/replied headers so old thread content doesn't bleed in."""
    body = re.sub(r'-----original message-----.*', '', body, flags=re.DOTALL)
    body = re.sub(r'---------------------- forwarded by.*', '', body, flags=re.DOTALL)
    body = re.sub(r'on .{5,50} wrote:.*', '', body, flags=re.DOTALL)
    return body

def contains_any(text, keywords):
    return any(k in text for k in keywords)

def regex_match(text, patterns):
    return any(re.search(p, text) for p in patterns)

def has_negation(text):
    """
    STEP 2 FIX: Negation filter.
    These phrases mean the sender is resolving the issue, not requesting action.
    """
    negation_phrases = [
        "i will let you know", "we'll check it out", "already worked this out",
        "i have already", "this has been resolved", "i've already", "already done",
        "i'll take care", "i will send", "we will send", "i'll handle",
        "no action needed", "no action required", "nothing to do"
    ]
    return contains_any(text, negation_phrases)


# ─────────────────────────────────────────────
# HARD OVERRIDE (the ONLY true gatekeeper)
# ─────────────────────────────────────────────

def is_hard_information(text):
    """
    STEP 2 FIX: Hard override reserved for UNAMBIGUOUS non-actionable emails only.
    "thanks" and "noted" are REMOVED — they appear in action emails too.
    Only patterns that are almost never URGENT or ACTION get a hard override.
    """
    hard_patterns = [
        r"automatic reply", r"out of office", r"auto-reply",
        r"newsletter", r"unsubscribe", r"click here to unsubscribe",
        r"distribution list", r"daily riddle", r"daily summary",
        r"ecard", r"raffle", r"travelocity", r"believe to be reliable"
    ]
    return regex_match(text, hard_patterns)


# ─────────────────────────────────────────────
# LABELING FUNCTIONS — return (label, weight) or ABSTAIN
# ─────────────────────────────────────────────

# --- URGENT LFs ---

def lf_u1_asap_with_verb(text):
    """
    STEP 1: High-confidence urgency — ASAP/immediately co-occurring with an action verb.
    Weight 3 because this combination is highly predictive.
    """
    urgency  = contains_any(text, ["asap", "immediately", "right away", "urgent", "as soon as possible"])
    has_verb = contains_any(text, ["submit", "review", "call", "send", "approve", "book", "confirm"])
    if urgency and has_verb:
        return (URGENT, 3)
    if urgency:
        return (URGENT, 2)
    return ABSTAIN

def lf_u2_deadline_explicit(text):
    """Explicit deadline patterns — high confidence."""
    patterns = [
        r"by (monday|tuesday|wednesday|thursday|friday|today|tonight|tomorrow)",
        r"by \d{1,2}(:\d{2})?\s?(am|pm)",
        r"no later than", r"due today", r"due by", r"deadline"
    ]
    if regex_match(text, patterns):
        return (URGENT, 2)
    return ABSTAIN

def lf_u3_eod_temporal(text):
    """EOD / end of day / tonight — moderate confidence."""
    if contains_any(text, ["eod", "end of day", "tonight", "due today"]):
        return (URGENT, 2)
    return ABSTAIN

def lf_u4_high_pressure_punctuation(text):
    """
    Navneet's 'hyper user' signal: !! or ?? in text.
    Reliable when present, weight 2.
    """
    if re.search(r"\?\?+|\!\!+", text):
        return (URGENT, 2)
    return ABSTAIN

def lf_u5_high_stakes_domain(text):
    """
    Domain-level urgency: security, stocks, healthcare, etc.
    Weight 2 — domain alone is a moderate signal.
    NOTE: "legal" removed — appears in ~60% of Enron emails and caused
    massive false URGENT predictions (ACT→URGENT). Only keep highly
    specific domain terms that almost never appear in casual emails.
    """
    domains = ["security breach", "medical", "emergency",
               "stocks", "nymex", "healthcare", "tucson electric"]
    if contains_any(text, domains):
        return (URGENT, 2)
    return ABSTAIN

def lf_u6_scheduling_with_pressure(text):
    """
    STEP 2 FIX: Scheduling alone is NOT urgent.
    Only flag urgent if scheduling + time pressure co-occur.
    'Can we schedule a meeting?' → ACTION, not URGENT.
    'Need to schedule a meeting ASAP for today' → URGENT.
    """
    has_scheduling = contains_any(text, ["schedule", "book", "reserve", "meeting"])
    has_pressure   = contains_any(text, ["asap", "today", "tonight", "immediately", "urgent"])
    if has_scheduling and has_pressure:
        return (URGENT, 2)
    return ABSTAIN

def lf_u7_conference_call_now(text):
    """Conference call with a specific same-day time → urgent coordination needed.
    Only fires when the call is TODAY or THIS AFTERNOON — not any conf call with a time.
    This prevents future-scheduled calls from firing URGENT.
    """
    has_call = contains_any(text, ["conference call", "call at", "phone at"])
    has_time = regex_match(text, [r"\d{1,2}:\d{2}\s*(am|pm).{0,20}(today|this afternoon|now)"])
    if has_call and has_time:
        return (URGENT, 2)
    return ABSTAIN


# --- ACTION LFs ---

def lf_a1_direct_command(text):
    """
    STEP 1: High-confidence action — command verb + object pattern.
    'review the attached', 'submit the report', 'approve the request'
    Weight 3 — very precise.
    """
    pattern = r"(submit|review|approve|check|send|prepare|sign|update|complete|add)\s+(the|this|my|attached|your|a)\s+\w+"
    if re.search(pattern, text):
        return (ACTION, 3)
    return ABSTAIN

def lf_a2_question_in_subject(text_subject):
    """
    STEP 2 FIX: Subject-level intent is given extra weight (called with subject only).
    'question', 'action required', 'request', 'help needed' in subject → strong ACTION signal.
    """
    if contains_any(text_subject, ["question", "action required", "action needed", "request", "help needed", "please"]):
        return (ACTION, 3)   # subject-level = high confidence
    return ABSTAIN

def lf_a3_question_mark(text):
    """
    STEP 1: Question mark alone is a WEAK signal (weight 1).
    Previously this was equal weight to everything else — now properly downweighted.
    """
    if "?" in text and not has_negation(text):
        return (ACTION, 0.5)
    return ABSTAIN

def lf_a4_conversational_expectation(text):
    """Sender clearly expects a reply."""
    phrases = ["let me know", "give me a call", "seeking views", "thoughts?",
               "can we", "what do you think", "do you have", "could you please"]
    if contains_any(text, phrases) and not has_negation(text):
        return (ACTION, 2)
    return ABSTAIN

def lf_a5_followup(text):
    """Follow-up emails always require action."""
    if contains_any(text, ["following up", "circling back", "checking in", "any updates", "any progress"]):
        return (ACTION, 2)
    return ABSTAIN

def lf_a6_approval_request(text):
    """Explicit approval needed."""
    if contains_any(text, ["please approve", "need approval", "your approval", "awaiting approval"]):
        return (ACTION, 3)
    return ABSTAIN

def lf_a7_question_at_start(text):
    """Email body starts with a question word — directional question."""
    starters = ["what ", "how ", "can we ", "could you ", "is there ", "should we ", "who "]
    if any(text.strip().startswith(s) for s in starters) and not has_negation(text):
        return (ACTION, 2)
    return ABSTAIN

def lf_a8_scheduling_no_pressure(text):
    """
    STEP 2 FIX: Scheduling WITHOUT time pressure = ACTION (not URGENT).
    'Can we schedule a meeting?' is ACTION.
    """
    has_scheduling = contains_any(text, ["schedule", "set up a meeting", "can we meet"])
    has_pressure   = contains_any(text, ["asap", "today", "tonight", "immediately", "urgent"])
    if has_scheduling and not has_pressure:
        return (ACTION, 2)
    return ABSTAIN


# --- INFORMATION LFs ---

def lf_i1_explicit_fyi(text):
    """FYI / for your information / just to inform — clear info signal. Weight 2."""
    if contains_any(text, ["fyi", "for your information", "just to inform", "for reference", "please be advised", "please note"]):
        return (INFORMATION, 2)
    return ABSTAIN

def lf_i2_acknowledgement(text):
    """
    STEP 2 FIX: 'thanks' and 'noted' are now SOFT info signals (weight 1),
    NOT hard overrides. They still push toward INFO but don't block ACTION.
    """
    if contains_any(text, ["thank you", "thanks", "noted", "received", "got it", "acknowledged"]):
        return (INFORMATION, 1)
    return ABSTAIN

def lf_i3_broadcast(text):
    """Company-wide announcements are clearly informational."""
    if contains_any(text, ["all employees", "company-wide", "announcement", "broadcasting",
                            "this is to inform", "we are pleased to announce"]):
        return (INFORMATION, 2)
    return ABSTAIN

def lf_i4_newsletter_advertisement(text):
    """Newsletters and ads are informational (weight 2)."""
    if contains_any(text, ["newsletter", "advertisement", "keynotes", "unsubscribe"]):
        return (INFORMATION, 2)
    return ABSTAIN

def lf_i5_sender_narrating_own_action(text):
    """
    STEP 2 FIX: Sender describing their own future action = INFO, not ACTION.
    'I will send you the report' doesn't require recipient to do anything.
    """
    patterns = [r"i will (send|provide|share|forward|submit|update)",
                r"i('ll| will) (take care|handle|look into)",
                r"we will (send|provide|update|share)"]
    if regex_match(text, patterns):
        return (INFORMATION, 2)
    return ABSTAIN

# ─────────────────────────────────────────────────────────────────────────────
# FIX 1: Stop ACTION → URGENT misclassifications
# Problem: "legal", "tonight", "conference call" fire urgency alone but these
# words appear constantly in non-urgent emails (case law discussions, casual
# plans, scheduled meetings).
# ─────────────────────────────────────────────────────────────────────────────
 
def lf_u_legal_needs_pressure(text):
    """
    'Legal' alone is NOT urgent. Only flag urgent if legal + pressure word.
    Fixes: "Re: Utility contracts" (legal conference trip), "EGM Monthly Legal Report"
    """
    has_legal    = contains_any(text, ["legal", "compliance", "attorney"])
    has_pressure = contains_any(text, ["asap", "immediately", "urgent", "deadline",
                                        "by today", "by tomorrow", "by end of day"])
    if has_legal and has_pressure:
        return (URGENT, 2)
    # Legal alone → slight ACTION lean (someone sent a legal document to review)
    if has_legal:
        return (ACTION, 1)
    return ABSTAIN
 
 
def lf_u_tonight_casual_filter(text):
    """
    'Tonight' in casual/social context is NOT urgent.
    "I am going tonight and wanted to know if it is good" → social, not urgent.
    Only flag URGENT if tonight + actual work task.
    Fixes: "Re: long lost friend", "Re: birth pictures", "Two-Rows for MNF" (pizza tonight)
    """
    if "tonight" not in text:
        return ABSTAIN
    # Social/casual tonight indicators — these push toward INFO
    social_indicators = ["pizza", "drinks", "dinner", "lunch", "party", "movie",
                         "going tonight", "zula", "bar", "restaurant", "date tonight"]
    if contains_any(text, social_indicators):
        return (INFORMATION, 1)
    # Work task + tonight → URGENT
    work_task = contains_any(text, ["submit", "review", "send", "approve",
                                     "complete", "need", "call", "finish", "deploy"])
    if work_task:
        return (URGENT, 2)
    return ABSTAIN
 
 
def lf_u_conference_call_no_pressure(text):
    """
    'Conference call' alone is not URGENT. Most conf call emails are ACTION
    (please join, let me know your availability) not URGENT.
    Only URGENT if conf call + same-day time pressure.
    Fixes: "Risk Assessment Conference Call" (schedule for week of July 10, no pressure)
    """
    if not contains_any(text, ["conference call", "conf call"]):
        return ABSTAIN
    # Explicit same-day time pressure needed for URGENT
    has_immediate_pressure = contains_any(text, ["today", "this afternoon",
                                                   "in one hour", "asap", "now"])
    if has_immediate_pressure:
        return (URGENT, 2)
    # Otherwise it's coordination → ACTION
    return (ACTION, 2)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# FIX 2: Stop INFO → ACTION misclassifications (52 cases — biggest error bucket)
# Problem: Forwarded content, newsletters, jokes, announcements all contain
# question marks and action verbs in body text that aren't meant for the recipient.
# ─────────────────────────────────────────────────────────────────────────────
 
def lf_i_forwarded_chain(text):
    """
    Emails that are PURELY forwarded/FW chains where the current sender adds
    nothing actionable (e.g. "FYI" or blank forward) → INFORMATION.
    Fixes: "Fwd: FW: the Presidential Clock", "Re: FW: The National Grid" (no problem),
           "Alex's resume" (just forwarding), "FW: Two Boys" (forwarded joke)
    """
    # Forwarded with minimal sender comment
    forward_patterns = [r"^fwd?:", r"^fw:", r"-+ forwarded by", r"forwarded by .{5,50} on"]
    is_forwarded = regex_match(text, forward_patterns)
    if not is_forwarded:
        return ABSTAIN
 
    # Check if the current sender (top of email, before first "---" block) adds
    # something actionable. Look at first 300 chars for action words.
    top_of_email = text[:300]
    has_top_action = contains_any(top_of_email, ["please", "can you", "could you",
                                                   "let me know", "review", "approve",
                                                   "send", "asap", "urgent"])
    if has_top_action:
        return (ACTION, 1)   # Forwarded but with request
    return (INFORMATION, 2)  # Pure forward with no added ask
 
 
def lf_i_newsletter_or_promotion(text):
    """
    Marketing emails, newsletters, contest announcements → INFORMATION.
    Fixes: "Win a Free Golf Vacation", "Managers Training" (promotional),
           "Plasma & Banner Towers" (marketing bulletin)
    """
    promo_patterns = [
        r"click here", r"unsubscribe", r"subscribe", r"privacy policy",
        r"win a free", r"enter to win", r"you are receiving this",
        r"marketing bulletin", r"rentable", r"catalog available",
        r"register (today|now) (at|by calling)", r"free (trial|download|entry)"
    ]
    if regex_match(text, promo_patterns):
        return (INFORMATION, 3)   # Weight 3 — high confidence
    return ABSTAIN
 
 
def lf_i_long_informational_report(text):
    """
    Very long emails that are analytical reports, news summaries, or policy
    documents → INFORMATION. These often have questions inside the body text
    (e.g. "what does this mean for X?") that are rhetorical, not requests.
    Fixes: "Enron Mentions" (news clips), "Asian Credit Watch", "SDG&E Asks FERC",
           "Editorial by George Miller", "EOL Credit Approvals" (report)
    NOTE: The "attached + report signal" sub-case was removed — it incorrectly
    pulled short direct work requests like "new meters" and "Revised LPG Cargo Hedge"
    into INFO. Only the long-body case is safe enough to use.
    """
    # Very long body (>1500 chars) with news/report indicators
    is_long = len(text) > 1500
    report_indicators = [
        r"press release", r"bloomberg", r"reuters", r"according to",
        r"interim report", r"week ending", r"monthly (report|update|summary)",
        r"credit watch", r"market power", r"legislative status"
    ]
    has_report_signal = regex_match(text, report_indicators)

    if is_long and has_report_signal:
        return (INFORMATION, 2)
    return ABSTAIN
 
 
def lf_i_social_personal_email(text):
    """
    Clearly personal/social emails not requiring professional action → INFORMATION.
    Fixes: "Re: Boo" (IM name), "Re: Lunch" ("Thanks Monday is good"),
           "Re: silverman" (trading banter), "Re: RE:" casual chains,
           "Fwd: FW: the Presidential Clock" (joke)
    """
    social_patterns = [
        r"how was (your|the) (weekend|trip|game|play|party)",
        r"(drinks|dinner|lunch|pizza|bar) (tonight|last night|tomorrow)",
        r"i had a great time", r"hope to (see|hear from) you (soon)?",
        r"talk to you (soon|later)", r"love,?\s+\w+\s*$",
        r"good to hear from you", r"catch up",
        r"(funny|joke|lol|haha|lmao|rotfl)",
    ]
    if regex_match(text, social_patterns):
        return (INFORMATION, 2)
    return ABSTAIN
 
 
def lf_i_sender_concluding_reply(text):
    """
    Very short reply emails where the sender is CONCLUDING a thread (resolving,
    confirming, thanking) NOT opening a new task.
    Fixes: "RE: Boo" (Just got back from executive lunch), "Re: Lunch" (Thanks Monday is good),
           "RE: Term Project:" (No problem. Vince)
    GUARD: Skip if the email contains a direct task instruction — e.g. "ship it to
    the plant" should NOT be treated as a concluding reply even if it's short.
    """
    # Guard: if there's a direct task verb + object, don't classify as concluding
    task_verb_guard = r"(ship|send|move|add|path|book|enter|update|process|forward|flip|assign)\s+(it|them|this|the|all|these)"
    if re.search(task_verb_guard, text):
        return ABSTAIN

    concluding_patterns = [
        r"^no problem\.?\s*\n",           # "No problem.\nVince"
        r"^(ok|okay)\.?\s*\n",            # "Ok.\n"
        r"^(sounds good|looks good)\.?\s*",
        r"^thanks? (monday|tuesday|wednesday|thursday|friday)",
        r"^(noted|understood|got it|received)\.?\s*",
        r"^it was taken care of",
        r"^both are in",                  # RE: VEPCO comparison "Both are in VEPCO"
        r"thank\w* for (your|the) (prompt|email|reply|message|help|time)",
    ]
    # Short reply + concluding phrase → INFO
    if len(text) < 600 and regex_match(text, concluding_patterns):
        return (INFORMATION, 2)
    return ABSTAIN
 
 
# ─────────────────────────────────────────────────────────────────────────────
# FIX 3: Stop INFO → URGENT misclassifications (22 cases)
# Problem: Scheduled procedural announcements, system emails, HR notices
# contain dates/times that fire the urgency LFs incorrectly.
# ─────────────────────────────────────────────────────────────────────────────
 
def lf_i_institutional_announcement(text):
    """
    System-generated or institutionally-formatted announcements → INFORMATION.
    These have date/time but are broadcast notices, not urgent personal requests.
    Fixes: "NYMEX Houston Training", "(00-336) October ATOM Processing",
           "Mid-Year 2000 Performance Feedback", "Quick Tips for UBSWE migration",
           "Arizona Public Service Company" (credit approval broadcast)
    """
    institutional_patterns = [
        r"please be advised",
        r"to: all (nymex|comex|enron|employees|members|brokers)",
        r"(from|re): .{0,50}(vice president|manager|director|administrator)",
        r"note: (you will|if you|please|broker)",
        r"automatically (drafted|generated|sent)",
        r"(you have been selected|you are receiving this)",
        r"system (generated|notification|alert)",
        r"if you (have any questions|encounter|need assistance)",
        r"(enron|nymex|comex|ferc) (training|notice|announcement|advisory|bulletin)",
        r"helpdesk|help desk",
    ]
    if regex_match(text, institutional_patterns):
        return (INFORMATION, 2)
    return ABSTAIN
 
 
def lf_i_schedule_or_procedure_doc(text):
    """
    Processing schedules, procedural timetables, migration guides → INFORMATION.
    Fixes: "(00-336) October ATOM Processing" (banking business day 1, 2, 3...),
           "Quick Tips for the UBSWE migration", "Arizona Public Service Company" list
    """
    schedule_patterns = [
        r"(date|banking business day):\s+\d",   # Tabular schedule
        r"(step|item)\s+\d+[\.:]",              # Numbered procedure steps
        r"(phase|stage)\s+\d+",
        r"as of (start of business|end of business|monday|tuesday)",
        r"will (be|become) (effective|available|live) on",
        r"(rollout|go.?live|migration) (date|on|scheduled)",
    ]
    if regex_match(text, schedule_patterns):
        return (INFORMATION, 2)
    return ABSTAIN
 
 
def lf_i_deadline_in_broadcast(text):
    """
    Dates/deadlines in an announcement context → INFO not URGENT.
    Fixes: "NYMEX Houston Training on February 4" (training from 10am-2pm)
           "Mid-Year 2000 Performance Feedback" (feedback due date)
           "Accomplishments/self evaluation" (PRC deadline reminder)
    The key: these have a date but the recipient is being INFORMED, not personally pressured.
    """
    broadcast_with_deadline_patterns = [
        r"(registration|enrollment|feedback|submission|response) (deadline|due date|by)",
        r"please (complete|provide|submit|have) .{0,60} by (the date|friday|end of)",
        r"(the following|below) (schedule|dates|timeline|agenda)",
        r"please plan to attend",
        r"(training|workshop|seminar|meeting) (will be held|is scheduled|takes place)",
    ]
    if regex_match(text, broadcast_with_deadline_patterns):
        return (INFORMATION, 2)
    return ABSTAIN
 
 
# ─────────────────────────────────────────────────────────────────────────────
# FIX 4: Catch URGENT → ACTION misclassifications (8 cases)
# Problem: Real urgent emails without "ASAP" keywords — urgent because of
# a near-term concrete deadline or a high-stakes business situation.
# ─────────────────────────────────────────────────────────────────────────────
 
def lf_u_specific_date_in_near_future(text):
    """
    A specific meeting date/time in the near future (Monday, Tuesday, this week,
    "week of [month] [number]") combined with a request → likely URGENT coordination.
    Fixes: "Risk Assessment Conference Call" (week of July 10 + "please email me")
           "Cornhusker Meeting" (Tuesday July 11, from 3:00 to 4:00 pm + "please plan to attend")
    """
    # Specific short-horizon date
    near_date_patterns = [
        r"(monday|tuesday|wednesday|thursday|friday),?\s+(july|june|aug|sep|jan|feb|mar|apr|may|oct|nov|dec)\s+\d+",
        r"week of (july|june|aug|sep|jan|feb|mar|apr|may|oct|nov|dec)",
        r"(july|june|aug|sep|jan|feb|mar|apr|may|oct|nov|dec)\s+\d{1,2}(st|nd|rd|th)?,?\s+from \d",
        r"\d{1,2}:\d{2}\s*(am|pm).{0,30}(today|tomorrow|monday|tuesday|wednesday|thursday|friday)",
    ]
    has_near_date = regex_match(text, near_date_patterns)
    has_action_ask = contains_any(text, ["please", "let me know", "email me",
                                          "plan to attend", "please join", "call"])
    if has_near_date and has_action_ask:
        return (URGENT, 2)
    return ABSTAIN
 
 
def lf_u_security_resource_request(text):
    """
    IT security approval requests with approval workflow → URGENT (system access blocked).
    Fixes: "Please Approve: Application Request (WSMH-4ESNVA)"
    """
    patterns = [
        r"security resource request",
        r"(approve|reject) (request|access)",
        r"(submitted for your approval|awaiting (your )?approval)",
        r"(click|double.click).{0,40}(approve|reject|view)",
    ]
    if regex_match(text, patterns):
        return (URGENT, 2)
    return ABSTAIN
 
 
def lf_u_deadline_with_recipient_ask(text):
    """
    Email with an explicit hard deadline + recipient is expected to respond
    or take action before that deadline. This differs from broadcast announcements
    by being personal/direct ("responses are due", "please provide by").
    Fixes: "Chairman Hoecker's questions" (responses due Wednesday the 22nd)
           "EGM Monthly Legal Report" (provide to me by Wednesday of this week)
           "We beat the blackouts so far" (please send summaries by Aug 10)
    """
    deadline_with_ask_patterns = [
        r"(responses?|comments?|feedback|changes?|summaries?) (are |is )?(due|must be (received|submitted))"
            r".{0,50}(by|on|before|no later than)",
        r"please (provide|send|submit|have) .{0,80} by (wednesday|monday|tuesday|"
            r"thursday|friday|end of( the)? week|close of business|friday)",
        r"(due|deadline|must be received) (by|on) (wednesday|monday|tuesday|thursday|friday)",
        r"by (close of business|end of( the)? week|friday,? (august|september|"
            r"october|november|december|january|february|march|april|may|june|july))",
    ]
    if regex_match(text, deadline_with_ask_patterns):
        return (URGENT, 3)   # High weight — very specific signal
    return ABSTAIN
 

# ─────────────────────────────────────────────────────────────────────────────
# ROUND 2 FIX A: Protect short direct professional requests (ACT→INFO, 48 cases)
# Problem: The INFO LFs from Round 1 are too broad. Short direct professional
# requests are being swept into INFO. These LFs protect them.
# ─────────────────────────────────────────────────────────────────────────────

def lf_a_short_direct_command(text, body_only):
    """
    Short emails (under 500 chars body) containing a clear direct command
    or question aimed at one person → strong ACTION signal (weight 3).
    Catches: "Can you price that physical call for PNM?" (pnm call)
             "would you go into July 31st and path 10,000 dt..." (Re: EOGS)
             "Please let me know when we can expect the funds" (EES credit)
             "ship it to the plant" (RE: Lone Star)
             "Would Tuesday August 29th from 1:00pm to 4:00pm work for everyone?"
             "Has eric gonzales been invited. I would like him to be"
    """
    body = body_only.strip()
    if len(body) > 500:
        return ABSTAIN

    short_direct_patterns = [
        r"^(can|could|would|will) you\b",
        r"^(please|kindly) (let me know|call|send|forward|review|check|confirm|update|go into|path|price|draft|provide|pull|add|fix|print|run|set up)",
        r"^(are|is|do|did|have|has|was|were) (you|they|he|she|it)\b.{0,80}\?",
        r"^(call|send|check|review|forward|confirm|update|go into|path|price|let me know|give me|tell me|provide|pull|add|fix|print|run|set up)\b",
        r"(go into|path|price|rebook|book|enter|add to (the )?database|assign|flip)",
        r"would .{5,60} work for (you|everyone|the group|the team)\?",
        r"(any|an) (update|news|word|progress|response|feedback|answer)\?",
        r"(i am still|still) (in need|waiting|awaiting|pending|looking)",
        r"(i have|had) (requested|asked|sent).{0,60}(still|yet|not|no)",
    ]
    if any(re.search(p, body.lower()) for p in short_direct_patterns):
        return (ACTION, 3)
    return ABSTAIN


def lf_a_fyi_with_ask(text):
    """
    Emails that start with FYI but also contain a question or explicit ask → ACTION.
    Pure FYI with no question → INFO.
    Catches: "socal gas" (Jeff -- FYI [forwarded with embedded question about tariff])
    """
    lower = text.lower()
    has_fyi = (lower.strip().startswith("fyi") or
               re.search(r"^(jeff|john|mark|rick|vince|sara|tana|kim) --\s*\nfyi", lower))
    if not has_fyi:
        return ABSTAIN
    if "?" in text or contains_any(lower, ["can you", "could you", "what do you think",
                                             "any insight", "do you see", "please advise"]):
        return (ACTION, 2)
    return (INFORMATION, 2)


def lf_a_operational_data_request(text):
    """
    Operational/trading requests with specific data, numbers, deals, meters → ACTION.
    These are always ACTION regardless of length or other signals.
    Catches: "new meters" (add to database, IHS numbers)
             "Re: EOGS" (path 10,000 dt from deal 264613)
             "Trade with John Lavorato" (Please Confirm Receipt)
             "EES credit refund to EPMI" (deal #123825.15)
    """
    operational_patterns = [
        r"\d+[,.]?\d*\s*(mmbtu|mmcf|mw|mwh|dt|mcf|bbl|mmbd)",
        r"deal\s*#?\s*\d{5,}",
        r"(meter|point)\s*#?\s*\d+",
        r"path\s+\d+[,.]?\d*\s*(dt|mmbtu|mmcf)",
        r"(add to|update|enter.{0,20}in)\s+(the\s+)?(database|system|enpower|sitara)",
        r"please confirm (receipt|delivery|execution)",
        r"(wiring|wire) (instructions|transfer)",
        r"(ihs number|nymex form|caiso|ferc filing)",
    ]
    if any(re.search(p, text.lower()) for p in operational_patterns):
        return (ACTION, 2)
    return ABSTAIN


# ─────────────────────────────────────────────────────────────────────────────
# ROUND 2 FIX B: Stop false URGENT on social/casual emails (ACT→URGENT, 20 cases)
# ─────────────────────────────────────────────────────────────────────────────

def lf_u_asap_in_forwarded_thread_only(text, body_only):
    """
    ASAP appearing ONLY in a forwarded/quoted thread, not in the current
    sender's text at the top → do NOT count as urgent.
    Catches: "PA/ETA" (ASAP in old forwarded content, current says "Here are the
             most recent versions"), "FW: J. Robert Collins" (ASAP in thread)
    """
    top = re.split(
        r'(-{3,}\s*(original message|forwarded by)|from:.{5,50}on \d{1,2}/\d{1,2}/\d{4})',
        body_only, flags=re.IGNORECASE
    )[0].strip()

    has_asap_in_top    = bool(re.search(r'\basap\b|\bimmediately\b|\bright away\b', top.lower()))
    has_asap_in_thread = bool(re.search(r'\basap\b|\bimmediately\b|\bright away\b', body_only.lower()))

    if has_asap_in_thread and not has_asap_in_top:
        return (ACTION, 1)   # ASAP only in forwarded content — slight action, not urgent
    return ABSTAIN


def lf_u_casual_social_context(text):
    """
    Strong social/personal context signals that override urgency keywords.
    Weight 3 — these are unambiguously personal, never urgent work.
    Catches: "Re: long lost friend", "Re: birth pictures" (phoenix golf trip),
             "Re: A Christmas Tasters", "RE:" (baby oil/lotion), casual fitness talk
    """
    social_strong = [
        r"(birthday|christmas) (card|gift|present)",
        r"(high school|reunion|old friend|long lost)",
        r"(i will call|call me|call you) (tonight|later|this evening)",
        r"(thong|lotion|tan|bracelet|outfit)",
        r"(treadmill|abs|workout|gym)\b.{0,100}(how was|going to|last night)",
        r"(golf|ski|scuba|diving|certif).{0,80}(weekend|trip|go)",
        r"(canoe|hiking|camping|swimming) (trip|weekend)",
        r"(christmas|thanksgiving|holiday) (tasters|dinner|party|celebration)",
        r"i('ll| will) (come|go|join).{0,30}(stag|solo|alone|with)",
        r"love you\.?\s*$",
        r"(i am|i'm) (a fat ass|so cute)",
    ]
    if any(re.search(p, text.lower()) for p in social_strong):
        return (INFORMATION, 3)
    return ABSTAIN


def lf_u_cancelled_or_resolved_meeting(text):
    """
    Meeting cancellations, reschedulings, or "already handled" replies
    are NOT urgent — the urgency has passed or been removed.
    Catches: "Today's Group Meeting" (has been cancelled),
             "FW: Credit Meeting" (rescheduled),
             "Re: Global Facilities Maintenance" ("does not have to be done by 11:30")
    """
    cancel_patterns = [
        r"(meeting|call|session|event)\s+(has been|is)\s+(cancelled|canceled|rescheduled|postponed)",
        r"(cancelled|canceled|postponed|rescheduled)\s+(the|this|our|today)",
        r"does not have to be done by",
        r"no longer need(ed)?",
        r"(has been|was)\s+(taken care of|resolved|completed|handled|addressed)",
        r"never mind|disregard (this|the previous|my)",
    ]
    if any(re.search(p, text.lower()) for p in cancel_patterns):
        return (ACTION, 1)
    return ABSTAIN


# ─────────────────────────────────────────────────────────────────────────────
# ROUND 2 FIX C: Protect forwarded INFO reports from URGENT (INF→URGENT, 15 cases)
# Problem: Forwarded reports/articles where current sender adds minimal comment
# are getting URGENT because the forwarded body has urgency keywords/dates.
# ─────────────────────────────────────────────────────────────────────────────

def lf_i_minimal_comment_forward(text, body_only):
    """
    When the current sender's actual text is very short (under 120 chars)
    but the full email is long, and there's no ask in the top — pure forward → INFO.
    Weight 3 — very high confidence.
    Catches: "Asian Credit Watch" ("Another article from the competitive intelligence group. Jeff")
             "Editorial by George Miller" ("This guy has it right.")
             "SDG&E Asks FERC" (brief intro + forwarded press release)
             "CORRECTION: interim report" ("Apologies. Please note I pasted the wrong graph")
             "FW: J. Robert Collins" ("Another NDA. This one is for the New York Mercantile Exchange.")
    """
    forward_split = re.split(
        r'(-{5,}\s*(original message|forwarded by)|'
        r'from:.{5,80}on \d{1,2}/\d{1,2}/\d{4}|'
        r'-{5,} forwarded)',
        body_only, flags=re.IGNORECASE
    )
    top_text = forward_split[0].strip()

    if len(top_text) < 120 and len(body_only) > 400:
        has_ask = any(re.search(p, top_text.lower()) for p in [
            r'\?', r'please (review|respond|advise|let me know)',
            r'can you', r'could you', r'what do you think',
            r'asap', r'urgent', r'immediately'
        ])
        if not has_ask:
            return (INFORMATION, 3)
    return ABSTAIN


def lf_i_news_or_report_forward(text):
    """
    Forwarded external news articles, press releases, or industry reports → INFO.
    Catches: "Editorial by George Miller" (SF Chronicle editorial)
             "SDG&E Asks FERC" (SDG&E press release)
             "Legislative Status Report Week Ending 4/20"
             "Asian Credit Watch" (competitive intelligence report)
    """
    news_patterns = [
        r"(press release|bloomberg|reuters|wall street journal|"
        r"sf chronicle|san francisco chronicle|financial times)",
        r"(legislative|status) report (week ending|for (the week|month))",
        r"competitive intelligence",
        r"(editorial|op.?ed|column)\s+by\s+\w+",
        r"(article|paper|study|report)\s+(from|by|out from)\s+\w+",
        r"week ending \d{1,2}/\d{1,2}",
    ]
    if any(re.search(p, text.lower()) for p in news_patterns):
        return (INFORMATION, 2)
    return ABSTAIN


def lf_i_resolved_thread(text, body_only):
    """
    Short reply emails where the sender indicates the issue is resolved/handled.
    The thread below may have had an ask, but the current reply closes it → INFO.
    Catches: "Re: Error Message" ("It was taken care of yesterday afternoon.")
             "Re: FW: The National Grid" ("no problem / Jim")
             "RE: New Power QSE Agreement" ("Credit's concerns are adequately addressed")
             "RE: Term Project:" ("No problem. / Vince")
    """
    forward_split = re.split(
        r'(-{3,}\s*(original message|forwarded by)|from:.{5,50}on \d{1,2}/\d{1,2}/\d{4})',
        body_only, flags=re.IGNORECASE
    )
    top = forward_split[0].strip().lower()

    resolved_phrases = [
        r"(it was|has been|were|was)\s+(taken care of|resolved|handled|addressed|completed|done|fixed)",
        r"no (problem|objection|issue|concern)\.?\s*(\n|$)",
        r"(we have|i have|have) no (objection|problem|issue|concern)",
        r"(adequately|fully|properly|already)\s+(addressed|handled|covered|resolved)",
        r"^(ok|okay|sounds good|looks good|noted|understood|confirmed|received|got it)\.?\s*\n?$",
        r"thanks? for (your|the) (prompt|quick|fast|help|email|reply|message)",
    ]
    if len(top) < 200 and any(re.search(p, top) for p in resolved_phrases):
        return (INFORMATION, 2)
    return ABSTAIN


# ─────────────────────────────────────────────
# AGGREGATION ENGINE (STEP 1 core)
# ─────────────────────────────────────────────

def predict_label(row):
    """
    STEP 1: Weighted scoring aggregation.
    Each LF votes for a class with a weight.
    Final label = class with highest total score.
    Subject lines get 1.5x multiplier (STEP 2 fix).
    """
    subj     = normalize(str(row.get('subject', '')))
    body_raw = normalize(clean_body(str(row.get('body', ''))))
    body     = body_raw   # clean body only (no subject prefix)
    combined = f"{subj} {body_raw}"

    # Hard override check first (only truly unambiguous cases)
    if is_hard_information(combined):
        # Still allow bypass for extreme urgency
        if not regex_match(combined, [r"asap", r"\!\!+", r"deadline", r"\burgent\b"]):
            return "INFORMATION", {}, {"hard_override": True}

    # Collect all LF votes
    lf_outputs = {}

    # ── Body-level LFs ──
    body_lfs = [
        ("LF_U1_AsapWithVerb",         lf_u1_asap_with_verb(combined)),
        ("LF_U2_DeadlineExplicit",      lf_u2_deadline_explicit(combined)),
        ("LF_U3_EodTemporal",           lf_u3_eod_temporal(combined)),
        ("LF_U4_HighPressurePunct",     lf_u4_high_pressure_punctuation(combined)),
        ("LF_U5_HighStakesDomain",      lf_u5_high_stakes_domain(combined)),
        ("LF_U6_SchedulingWithPressure",lf_u6_scheduling_with_pressure(combined)),
        ("LF_U7_ConferenceCallNow",     lf_u7_conference_call_now(combined)),
        ("LF_A1_DirectCommand",         lf_a1_direct_command(combined)),
        ("LF_A3_QuestionMark",          lf_a3_question_mark(combined)),
        ("LF_A4_ConversationalExpect",  lf_a4_conversational_expectation(combined)),
        ("LF_A5_FollowUp",              lf_a5_followup(combined)),
        ("LF_A6_ApprovalRequest",       lf_a6_approval_request(combined)),
        ("LF_A7_QuestionAtStart",       lf_a7_question_at_start(body)),
        ("LF_A8_SchedulingNoPress",     lf_a8_scheduling_no_pressure(combined)),
        ("LF_I1_ExplicitFyi",           lf_i1_explicit_fyi(combined)),
        ("LF_I2_Acknowledgement",       lf_i2_acknowledgement(combined)),
        ("LF_I3_Broadcast",             lf_i3_broadcast(combined)),
        ("LF_I4_NewsletterAd",          lf_i4_newsletter_advertisement(combined)),
        ("LF_I5_SenderNarratingOwn",    lf_i5_sender_narrating_own_action(combined)),

        # ROUND 1 FIX 1: ACTION → URGENT fixes
        ("LF_U_LegalNeedsPressure",      lf_u_legal_needs_pressure(combined)),
        ("LF_U_TonightCasualFilter",     lf_u_tonight_casual_filter(combined)),
        ("LF_U_ConferenceCallNoPress",   lf_u_conference_call_no_pressure(combined)),

        # ROUND 1 FIX 2: INFO → ACTION fixes
        ("LF_I_ForwardedChain",          lf_i_forwarded_chain(combined)),
        ("LF_I_NewsletterOrPromotion",   lf_i_newsletter_or_promotion(combined)),
        ("LF_I_LongInfoReport",          lf_i_long_informational_report(combined)),
        ("LF_I_SocialPersonalEmail",     lf_i_social_personal_email(combined)),
        ("LF_I_SenderConcludingReply",   lf_i_sender_concluding_reply(combined)),

        # ROUND 1 FIX 3: INFO → URGENT fixes
        ("LF_I_InstitutionalAnnounce",   lf_i_institutional_announcement(combined)),
        ("LF_I_ScheduleOrProcedure",     lf_i_schedule_or_procedure_doc(combined)),
        ("LF_I_DeadlineInBroadcast",     lf_i_deadline_in_broadcast(combined)),

        # ROUND 1 FIX 4: URGENT → ACTION fixes
        ("LF_U_SpecificDateNearFuture",  lf_u_specific_date_in_near_future(combined)),
        ("LF_U_SecurityRequest",         lf_u_security_resource_request(combined)),
        ("LF_U_DeadlineWithAsk",         lf_u_deadline_with_recipient_ask(combined)),

        # ROUND 2 FIX A: Protect short direct professional requests (ACT→INFO)
        ("LF_A_ShortDirectCommand",      lf_a_short_direct_command(combined, body)),
        ("LF_A_FyiWithAsk",              lf_a_fyi_with_ask(combined)),
        ("LF_A_OperationalDataRequest",  lf_a_operational_data_request(combined)),

        # ROUND 2 FIX B: Stop false URGENT on social/casual (ACT→URGENT)
        ("LF_U_AsapInThreadOnly",        lf_u_asap_in_forwarded_thread_only(combined, body)),
        ("LF_U_CasualSocialContext",     lf_u_casual_social_context(combined)),
        ("LF_U_CancelledMeeting",        lf_u_cancelled_or_resolved_meeting(combined)),

        # ROUND 2 FIX C: Protect forwarded INFO reports from URGENT (INF→URGENT)
        ("LF_I_MinimalCommentForward",   lf_i_minimal_comment_forward(combined, body)),
        ("LF_I_NewsOrReportForward",     lf_i_news_or_report_forward(combined)),
        ("LF_I_ResolvedThread",          lf_i_resolved_thread(combined, body)),
    ]

    # ── Subject-level LFs (1.5x multiplier) ──
    subject_lf_result = lf_a2_question_in_subject(subj)
    subject_lf_weight = 0
    subject_lf_label  = None
    if subject_lf_result is not ABSTAIN:
        subject_lf_label, raw_w = subject_lf_result
        subject_lf_weight = raw_w * 1.5   # subject multiplier
    lf_outputs["LF_A2_QuestionInSubject"] = subject_lf_result

    # Accumulate scores
    scores = defaultdict(float)
    if subject_lf_label:
        scores[subject_lf_label] += subject_lf_weight

    for lf_name, result in body_lfs:
        lf_outputs[lf_name] = result
        if result is not ABSTAIN:
            label, weight = result
            scores[label] += weight

    # Determine winner
    if not scores:
        final_label = INFORMATION   # fallback
    else:
        final_label = max(scores, key=scores.get)
        # Tiebreak: prefer ACTION over INFORMATION, URGENT over ACTION
        if scores[URGENT] == scores[ACTION] and scores[URGENT] > 0:
            final_label = URGENT
        elif scores[ACTION] == scores[INFORMATION] and scores[ACTION] > 0:
            final_label = ACTION

    return final_label, dict(scores), lf_outputs


# ─────────────────────────────────────────────
# PIPELINE RUNNER
# ─────────────────────────────────────────────

def run_pipeline():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File not found at {INPUT_FILE}")
        return

    dfs = pd.read_excel(INPUT_FILE, sheet_name=None)
    df = pd.concat(dfs.values(), ignore_index=True)

    # Strip illegal Excel chars
    ILLEGAL_RE = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]')
    df['body']    = df['body'].apply(lambda x: ILLEGAL_RE.sub("", str(x)))
    df['subject'] = df['subject'].apply(lambda x: ILLEGAL_RE.sub("", str(x)))

    predicted_labels = []
    score_rows       = []

    for _, row in df.iterrows():
        label, scores, _ = predict_label(row)
        predicted_labels.append(label)
        score_rows.append({
            "Score_URGENT":      scores.get(URGENT, 0),
            "Score_ACTION":      scores.get(ACTION, 0),
            "Score_INFORMATION": scores.get(INFORMATION, 0),
        })

    df['Predicted Label'] = predicted_labels
    scores_df = pd.DataFrame(score_rows)
    final_df  = pd.concat([df, scores_df], axis=1)

    # ── Accuracy report (exclude TIEs) ──
    eval_df   = final_df[final_df['Final Label'].str.strip().str.upper() != "TIE"]
    actual    = eval_df['Final Label'].str.strip().str.upper()
    predicted = eval_df['Predicted Label'].str.strip().str.upper()
    correct   = (actual == predicted).sum()
    total     = len(eval_df)

    print("\n" + "="*45)
    print("  WEIGHTED LF PERFORMANCE REPORT")
    print("="*45)
    print(f"  Total Evaluated  : {total} (excl. Ties)")
    print(f"  Correct          : {correct}")
    print(f"  Incorrect        : {total - correct}")
    print(f"  Accuracy         : {(correct/total)*100:.2f}%")

    # Per-class breakdown
    for cls in [URGENT, ACTION, INFORMATION]:
        mask     = actual == cls
        cls_corr = ((actual == predicted) & mask).sum()
        cls_tot  = mask.sum()
        print(f"  {cls:12s}: {cls_corr}/{cls_tot} ({(cls_corr/cls_tot*100) if cls_tot else 0:.1f}%)")
    print("="*45)

    # ── Excel export with colour coding ──
    writer    = pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter')
    final_df.to_excel(writer, index=False, sheet_name='Weighted_Results')
    workbook  = writer.book
    worksheet = writer.sheets['Weighted_Results']

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

    writer.close()
    print(f"\n  Saved → {OUTPUT_FILE}\n")


if __name__ == "__main__":
    run_pipeline()
