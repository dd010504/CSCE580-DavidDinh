import os
import cv2

# --- CONFIGURATION ---
# Input folder containing your renamed images (class_01.jpg, etc.)
INPUT_DIR = "raw_images"
# Output folder for the clean, resized images
OUTPUT_DIR = "processed_images"
# Target width in pixels (Standardizes size for any potential model)
TARGET_WIDTH = 1024

def main():
    # 1. Create Output Directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # 2. Get list of images
    # We look for .jpg, .jpeg, .png
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    files.sort() # Ensure we process in order (class_01, class_02...)
    
    print(f"Found {len(files)} images in '{INPUT_DIR}'")

    for filename in files:
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        # 3. Read Image
        img = cv2.imread(input_path)
        
        if img is None:
            print(f"[WARNING] Could not read {filename}. Skipping.")
            continue
            
        # 4. Resize Logic
        # Calculate new height to maintain aspect ratio
        height, width = img.shape[:2]
        scale_factor = TARGET_WIDTH / float(width)
        new_height = int(height * scale_factor)
        
        # Resize using INTER_AREA (best for shrinking)
        img_resized = cv2.resize(img, (TARGET_WIDTH, new_height), interpolation=cv2.INTER_AREA)
        
        # 5. Save Processed Image
        cv2.imwrite(output_path, img_resized)
        print(f"[OK] Processed: {filename} ({width}x{height} -> {TARGET_WIDTH}x{new_height})")

    print("\nPreprocessing complete. Images are ready in 'processed_images/'.")

if __name__ == "__main__":
    main()