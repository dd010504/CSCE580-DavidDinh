import os
import cv2

# --- CONFIGURATION ---
INPUT_DIR = "../raw_images"
OUTPUT_DIR = "../processed_images"
TARGET_WIDTH = 1024
MAKE_BINARY = True  # Set to True for High-Contrast Black & White

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"ERROR: Could not find input folder at: {os.path.abspath(INPUT_DIR)}")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    files.sort()
    
    print(f"Processing {len(files)} images...")

    for filename in files:
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        # 1. Read
        img = cv2.imread(input_path)
        if img is None: continue
            
        # 2. Resize
        height, width = img.shape[:2]
        scale_factor = TARGET_WIDTH / float(width)
        new_height = int(height * scale_factor)
        img_resized = cv2.resize(img, (TARGET_WIDTH, new_height), interpolation=cv2.INTER_AREA)
        
        # 3. Optional: Convert to "Scanned" look (Black & White)
        final_img = img_resized
        if MAKE_BINARY:
            # Convert to grayscale
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            # Apply slight noise removal (blur) then threshold
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            # Adaptive threshold handles shadows/uneven lighting well
            final_img = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )

        # 4. Save
        cv2.imwrite(output_path, final_img)
        print(f"[OK] Processed: {filename} (Resized to {TARGET_WIDTH}px width)")

    print("\nDone! Check 'processed_images' for black & white versions.")

if __name__ == "__main__":
    main()