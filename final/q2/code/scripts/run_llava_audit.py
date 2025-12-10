import ollama
import os
import pandas as pd
from fuzzywuzzy import process

# --- CONFIGURATION ---
IMAGE_DIR = "../processed_images"
ROSTER_FILE = "../data/roster.csv"

def get_valid_users():
    if not os.path.exists(ROSTER_FILE): return []
    df = pd.read_csv(ROSTER_FILE, header=None, names=['id', 'name', 'user'])
    return df['user'].astype(str).str.strip().str.lower().tolist()

def main():
    print("--- Starting LLaVA (Visual LLM) Attendance Audit ---")
    
    # 1. Setup Roster
    valid_users = get_valid_users()
    print(f"Loaded {len(valid_users)} valid usernames from roster.")

    # 2. Get Images
    if not os.path.exists(IMAGE_DIR):
        print("Error: Processed images folder not found.")
        return
    images = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')])
    
    results = []
    
    print("-" * 60)
    print(f"{'Class':<10} | {'LLaVA Raw Output (First 50 chars)':<40}")
    print("-" * 60)

    for img_name in images:
        class_key = img_name.replace(".jpg", "")
        img_path = os.path.join(IMAGE_DIR, img_name)
        
        # 3. PROMPT LLaVA
        # We give it a specific instruction to act like a transcriber.
        prompt = "Read this attendance sheet. List ONLY the handwritten usernames or signatures visible. Return them as a comma-separated list. Do not include dates or headers."
        
        try:
            response = ollama.generate(
                model='llava', 
                prompt=prompt, 
                images=[img_path]
            )
            raw_text = response['response']
            
            # 4. Process Output (Clean & Match)
            # LLaVA is chatty, so we try to split by commas or newlines
            potential_names = re.split(r'[,\n]', raw_text)
            
            found_students = set()
            for name in potential_names:
                clean = name.strip().lower()
                if len(clean) < 3: continue
                
                # Fuzzy match against roster
                match, score = process.extractOne(clean, valid_users)
                if score > 80:
                    found_students.add(match)
            
            print(f"{class_key:<10} | {raw_text[:50].replace(chr(10), ' '):<40}...")
            
            results.append({
                'Class': class_key,
                'Raw_LLaVA': raw_text,
                'Matched_Count': len(found_students),
                'Students': ", ".join(found_students)
            })
            
        except Exception as e:
            print(f"Error processing {class_key}: {e}")

    # Save
    df = pd.DataFrame(results)
    df.to_csv("llava_results.csv", index=False)
    print("\n[DONE] LLaVA results saved to 'llava_results.csv'")

if __name__ == "__main__":
    import re
    main()