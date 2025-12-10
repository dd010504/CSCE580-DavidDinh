import easyocr
import os
import pandas as pd
import re
from fuzzywuzzy import process

# --- CONFIGURATION ---
IMAGE_DIR = "../processed_images"
ROSTER_FILE = "../data/roster.csv"
GROUND_TRUTH_FILE = "../data/attendance_data.csv"
OUTPUT_FILE = "ai_generated_attendance.csv"

def load_ground_truth():
    """Loads the manual CSV to compare against AI results (for accuracy checking)."""
    if not os.path.exists(GROUND_TRUTH_FILE):
        return {}
    
    df = pd.read_csv(GROUND_TRUTH_FILE)
    raw_list = [df.columns[0]] + df.iloc[:, 0].tolist()
    
    gt_map = {} # { "class_01": {"student1", "student2"} }
    current_class = None
    
    for item in raw_list:
        text = str(item).strip().lower()
        if "class" in text:
            # Extract simple class key "class_01"
            parts = text.split(',')
            current_class = parts[0].strip().replace(" ", "_")
            gt_map[current_class] = set()
        elif text not in ['nan', 'adocteur', '']:
            if current_class:
                gt_map[current_class].add(text)
    return gt_map

def main():
    print("--- Initializing AI Model (EasyOCR) ---")
    # 'en' for English. gpu=False ensures it runs even without a dedicated graphics card.
    reader = easyocr.Reader(['en'], gpu=True) 
    
    # Load Roster for matching
    print("Loading Roster...")
    if not os.path.exists(ROSTER_FILE):
        print("Error: Roster file missing.")
        return
        
    df_roster = pd.read_csv(ROSTER_FILE, header=None, names=['id', 'name', 'user'])
    valid_usernames = df_roster['user'].astype(str).str.strip().str.lower().tolist()
    
    # Load Ground Truth for validation
    ground_truth_map = load_ground_truth()
    
    # Get Images
    images = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')])
    results = []
    
    print(f"Starting OCR on {len(images)} images...")
    print("-" * 60)
    print(f"{'Class':<10} | {'Detected':<10} | {'Valid Matches':<15} | {'Actual (GT)':<12} | {'Accuracy':<10}")
    print("-" * 60)

    for img_name in images:
        class_key = img_name.replace(".jpg", "") # class_01
        img_path = os.path.join(IMAGE_DIR, img_name)
        
        # 1. RUN AI MODEL (The heavy lifting)
        # detail=0 gives simple list of text
        try:
            detections = reader.readtext(img_path, detail=0)
        except Exception as e:
            print(f"Error reading {img_name}: {e}")
            continue
            
        # 2. POST-PROCESSING (Fuzzy Logic)
        # The AI sees "j0hn" -> we match to "john"
        found_students = set()
        for word in detections:
            clean_word = word.lower().strip()
            # Skip short noise
            if len(clean_word) < 3: continue
            
            # Fuzzy match against roster (Threshold 85/100)
            match, score = process.extractOne(clean_word, valid_usernames)
            if score > 80:
                found_students.add(match)
        
        # 3. ACCURACY CHECK
        # Compare what AI found vs Manual CSV (Ground Truth)
        actual_students = ground_truth_map.get(class_key, set())
        
        # Calculate Intersection (How many AI got right)
        correct_identifications = found_students.intersection(actual_students)
        
        # Precision: Correct / Total Found
        # Recall: Correct / Total Actual
        if len(actual_students) > 0:
            accuracy = (len(correct_identifications) / len(actual_students)) * 100
        else:
            accuracy = 0.0
            
        print(f"{class_key:<10} | {len(detections):<10} | {len(found_students):<15} | {len(actual_students):<12} | {accuracy:.1f}%")
        
        results.append({
            'Class': class_key,
            'AI_Raw_Count': len(detections),
            'AI_Matched_Count': len(found_students),
            'Ground_Truth_Count': len(actual_students),
            'Accuracy': accuracy
        })

    # Save Results
    df_res = pd.DataFrame(results)
    df_res.to_csv("ai_ocr_results.csv", index=False)
    print("-" * 60)
    print(f"Average Model Accuracy: {df_res['Accuracy'].mean():.2f}%")
    print("Detailed results saved to 'ai_ocr_results.csv'")

if __name__ == "__main__":
    main()