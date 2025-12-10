import pandas as pd
import re

# --- CONFIGURATION ---
# Ensure your manual attendance file is renamed to this
ATTENDANCE_FILE = '../data/attendance_data.csv'
# Ensure your roster file is renamed to this
ROSTER_FILE = '../data/roster.csv'

def main():
    print("--- Starting Attendance Audit ---")

    # 1. LOAD DATA
    try:
        # Load Attendance: We treat the header as the first data point to capture "class_01"
        df_att = pd.read_csv(ATTENDANCE_FILE)
        # Flatten the single column into a list, including the header itself
        raw_list = [df_att.columns[0]] + df_att.iloc[:, 0].tolist()
        
        # Load Roster: No header, columns are ID, Name, Username
        df_roster = pd.read_csv(ROSTER_FILE, header=None, names=['id', 'name', 'user'])
        # Create a set of valid students for O(1) lookup
        roster_users = set(df_roster['user'].astype(str).str.strip().str.lower())
        print(f"Roster loaded: {len(roster_users)} students.")
        
    except Exception as e:
        print(f"CRITICAL ERROR: Could not read files. {e}")
        print("Ensure 'attendance_data.csv' and 'roster.csv' are in this folder.")
        return

    # 2. PARSE ATTENDANCE LOGS
    # Data structure: { "class_01": ["student1", "student2"...] }
    attendance_map = {}
    class_dates = {}
    current_class = None
    
    # Regex to find "class_XX" and optional date "MM/DD/YYYY"
    # Matches: "class_01", "class_01, 8/19/2025", "Class 01"
    header_pattern = re.compile(r"class[_\s](\d+)(?:,\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4}))?", re.IGNORECASE)

    for item in raw_list:
        text = str(item).strip()
        text_lower = text.lower()
        
        # Check if this line is a Class Header
        match = header_pattern.search(text)
        if match:
            # Extract Class Number (e.g., "01")
            class_num = int(match.group(1))
            class_key = f"class_{class_num:02d}"
            
            # Extract Date if present (e.g., "8/19/2025")
            date_str = match.group(2) if match.group(2) else "Unknown"
            
            current_class = class_key
            attendance_map[current_class] = []
            class_dates[current_class] = date_str
            
        elif text_lower not in ['nan', 'adocteur', '']:
            # It is a student username
            if current_class:
                attendance_map[current_class].append(text_lower)

    # 3. ANALYZE STATISTICS
    results = []
    all_seen_students = set()

    for cls_key, students in attendance_map.items():
        unique_students = set(students)
        all_seen_students.update(unique_students)
        
        # Check against Key Dates from Exam Prompt (Page 3 Reference)
        date_str = class_dates.get(cls_key, "")
        event_note = ""
        
        # Correlation checks 
        if "10/7" in date_str or "10-07" in date_str: 
            event_note = " (Quiz 2)"
        elif "11/11" in date_str or "11-11" in date_str: 
            event_note = " (Quiz 3)"
        elif "11/18" in date_str or "11-18" in date_str: 
            event_note = " (Paper Pres)"
            
        results.append({
            'Class': cls_key,
            'Date': date_str + event_note,
            'Count': len(unique_students)
        })

    # Create DataFrame for clean sorting/calc
    df_res = pd.DataFrame(results)
    
    # Extract numeric value from "class_XX" for proper sorting
    df_res['SortNum'] = df_res['Class'].str.extract(r'(\d+)').astype(float)
    df_res = df_res.sort_values('SortNum').drop(columns=['SortNum'])

    # 4. CALCULATE Q2c ANSWERS
    # Median is calculated on classes that actually happened
    held_classes = df_res[df_res['Count'] > 0]
    median_val = held_classes['Count'].median()
    
    # Highest and Lowest
    highest = df_res.loc[df_res['Count'].idxmax()]
    lowest = held_classes.loc[held_classes['Count'].idxmin()]
    
    # Dropouts: Students in attendance logs who are NOT in final roster
    dropped_students = sorted(list(all_seen_students - roster_users))

    # 5. GENERATE REPORT
    print("\n" + "="*50)
    print("        FINAL CLASS ATTENDANCE REPORT")
    print("="*50)
    print(df_res.to_string(index=False))
    print("-" * 50)
    
    print(f"Q2c RESULTS:")
    print(f"1. Number of Classes Found: {len(df_res)}")
    print(f"2. Median Attendance:       {median_val}")
    print(f"3. Highest Attendance:      {highest['Count']} on {highest['Date']}")
    print(f"4. Lowest Attendance:       {lowest['Count']} on {lowest['Date']}")
    
    print("-" * 50)
    print(f"Correlation Analysis:")
    quiz2 = df_res[df_res['Date'].str.contains("Quiz 2", na=False)]
    if not quiz2.empty:
        print(f"-> Quiz 2 Attendance: {quiz2.iloc[0]['Count']} (vs Median {median_val})")
        
    quiz3 = df_res[df_res['Date'].str.contains("Quiz 3", na=False)]
    if not quiz3.empty:
        print(f"-> Quiz 3 Attendance: {quiz3.iloc[0]['Count']} (Note: 0 indicates no sheet used)")

    print("-" * 50)
    print(f"Students who dropped ({len(dropped_students)}):")
    print(", ".join(dropped_students))
    
    # Save to CSV for the "Finals" folder requirement
    df_res.to_csv('final_report_table.csv', index=False)
    print("\n[SUCCESS] Report saved to 'final_report_table.csv'")

if __name__ == "__main__":
    main()