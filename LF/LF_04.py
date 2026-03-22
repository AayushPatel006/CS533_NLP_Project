import pandas as pd
import re
import os

#Total Emails Evaluated : 97 (excluding Ties)
#Correctly Predicted    : 56
#Incorrectly Predicted  : 41
#Accuracy Score         : 57.73%


def normalize(text):
    if not text or not isinstance(text, str): 
        return ""
    return re.sub(r"\s+", " ", text.lower()).strip()

def analyze_email(row):
    """
    Applies the Enron-specific LFs and returns a dictionary of triggered 
    sub-labels and the final predicted label.
    """
    subj = normalize(str(row.get('subject', '')))
    body = normalize(str(row.get('body', '')))
    
    # Clean out forwarded message headers to avoid analyzing previous thread noise
    body_clean = re.sub(r'-----original message-----.*', '', body, flags=re.DOTALL)
    body_clean = re.sub(r'---------------------- forwarded by.*', '', body_clean, flags=re.DOTALL)
    
    text = f"{subj} {body_clean}"
    
    # Initialize LF tracking
    lfs = {
        "LF_Info_Gatekeeper": 0,
        "LF_Urgent_Trigger": 0,
        "LF_Action_Trigger": 0
    }
    
    # ---------------------------------------------------------
    # 1. GATEKEEPER: INFORMATION OVERRIDES
    # ---------------------------------------------------------
    info_overrides = [
        r"please be advised", r"fyi", r"for your information", r"newsletter", 
        r"advertisement", r"broadcasting", r"announcement", r"all employees", 
        r"distribution list", r"automatic reply", r"keynotes", r"thank you", 
        r"thanks", r"noted", r"received", r"believe to be reliable", r"daily summary", 
        r"interim report", r"ecard", r"raffle", r"travelocity", r"daily riddle", r"already worked this out"
    ]
    if any(re.search(pattern, text) for pattern in info_overrides):
        # Allow bypass ONLY for extreme urgency
        if not re.search(r"asap|\!\!+|deadline|urgent", text):
            lfs["LF_Info_Gatekeeper"] = 1

    # ---------------------------------------------------------
    # 2. URGENT TRIGGERS
    # ---------------------------------------------------------
    urgent_patterns = [
        r"asap", r"immediately", r"urgent", r"tonight", r"due today", r"right away",
        r"security resource request", r"book the hotel", r"visiting san francisco",
        r"\d{1,2}:\d{2}\s*(am|pm)", r"deadline", r"conference call", r"week ending", 
        r"\!\!+", r"\?\?+" 
    ]
    if any(re.search(pattern, text) for pattern in urgent_patterns):
        lfs["LF_Urgent_Trigger"] = 1

    # ---------------------------------------------------------
    # 3. ACTION TRIGGERS
    # ---------------------------------------------------------
    action_patterns = [
        r"\?", r"can you", r"could you", r"would you", r"let me know", r"action required",
        r"we should", r"assistance is requested", r"give me a call", r"seeking views",
        r"forward to all", r"review the", r"submit the", r"approve the", r"send to me", 
        r"check this", r"update me", r"add to the database", r"checking to see",
        r"i support this", r"please \w+"
    ]
    if any(re.search(pattern, text) for pattern in action_patterns):
        if "i will let you know" not in text and "we'll check it out" not in text:
            lfs["LF_Action_Trigger"] = 1

    # ---------------------------------------------------------
    # FINAL PREDICTION LOGIC
    # ---------------------------------------------------------
    if lfs["LF_Info_Gatekeeper"] == 1:
        predicted = "INFORMATION"
    elif lfs["LF_Urgent_Trigger"] == 1:
        predicted = "URGENT"
    elif lfs["LF_Action_Trigger"] == 1:
        predicted = "ACTION"
    else:
        predicted = "INFORMATION"
        
    lfs["Predicted Label"] = predicted
    return pd.Series(lfs)

def run_evaluation_pipeline(input_filepath, output_filepath):
    """
    Reads the dataset, applies the NLP logic, prints terminal stats, 
    and exports a color-coded Excel file.
    """
    if not os.path.exists(input_filepath):
        print(f"Error: Could not find file at {input_filepath}")
        return

    print(f"Loading data from: {input_filepath}...")
    df = pd.read_csv(input_filepath)
    
    # Strip illegal Excel characters from the text to prevent crashes
    ILLEGAL_RE = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]')
    df['body'] = df['body'].apply(lambda x: ILLEGAL_RE.sub("", str(x)))
    df['subject'] = df['subject'].apply(lambda x: ILLEGAL_RE.sub("", str(x)))

    # Apply the analysis function
    analysis_results = df.apply(analyze_email, axis=1)
    
    # Combine original data with the new analysis columns
    final_df = pd.concat([df, analysis_results], axis=1)

    # --- TERMINAL ACCURACY REPORT ---
    # Filter out 'TIE' rows so they don't skew the true algorithm performance
    eval_df = final_df[final_df['Final Label'].str.strip().str.upper() != "TIE"]
    
    actual = eval_df['Final Label'].str.strip().str.upper()
    predicted = eval_df['Predicted Label'].str.strip().str.upper()
    
    correct = (actual == predicted).sum()
    total = len(eval_df)
    accuracy = (correct / total) * 100

    print("\n" + "="*40)
    print("🎯 ALGORITHM PERFORMANCE REPORT")
    print("="*40)
    print(f"Total Emails Evaluated : {total} (excluding Ties)")
    print(f"Correctly Predicted    : {correct}")
    print(f"Incorrectly Predicted  : {total - correct}")
    print(f"Accuracy Score         : {accuracy:.2f}%")
    print("="*40 + "\n")

    # --- EXCEL EXPORT & FORMATTING ---
    print(f"Exporting results to Excel...")
    writer = pd.ExcelWriter(output_filepath, engine='xlsxwriter')
    final_df.to_excel(writer, index=False, sheet_name='Labeled_Results')
    
    workbook = writer.book
    worksheet = writer.sheets['Labeled_Results']
    
    # Define styles for correct (Green) and incorrect (Red)
    format_correct = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
    format_incorrect = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
    
    # Find column indices
    col_final = final_df.columns.get_loc("Final Label")
    col_pred = final_df.columns.get_loc("Predicted Label")
    
    # Apply formatting row by row
    for row_idx in range(1, len(final_df) + 1):
        actual_val = str(final_df.iloc[row_idx-1, col_final]).strip().upper()
        pred_val = str(final_df.iloc[row_idx-1, col_pred]).strip().upper()
        
        # Skip coloring if it was a TIE
        if actual_val == "TIE":
            continue
            
        cell_format = format_correct if actual_val == pred_val else format_incorrect
        worksheet.write(row_idx, col_pred, final_df.iloc[row_idx-1, col_pred], cell_format)
        
    writer.close()
    print(f"✅ Success! Color-coded analysis saved to: {output_filepath}")

# --- EXECUTION ---
if __name__ == "__main__":
    # Define your specific inputs and outputs here
    INPUT_CSV = "/Users/chandan/Desktop/NLP/Golden Dataset - 300 rows.xlsx - Batch 3.csv"
    OUTPUT_EXCEL = "/Users/chandan/Desktop/NLP/Batch3_Evaluated.xlsx"
    
    run_evaluation_pipeline(INPUT_CSV, OUTPUT_EXCEL)