import os
import cv2  # OpenCV
import shutil

# Paths
RAW_DIR = "../raw_images"
PROCESSED_DIR = "../processed_images"

# Create processed directory if it doesn't exist
if not os.path.exists(PROCESSED_DIR):
    os.makedirs(PROCESSED_DIR)

def preprocess_images():
    # Get all files
    files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    files.sort()  # Sort to keep order consistent

    print(f"Found {len(files)} images.")

    for i, filename in enumerate(files):
        img_path = os.path.join(RAW_DIR, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Could not read {filename}")
            continue

        # 1. Resize if too large (LLMs/Vision models often choke on 4k+ images)
        # We limit max dimension to 1024px or 2048px to save speed/memory
        height, width = img.shape[:2]
        max_dim = 1500 
        
        if max(height, width) > max_dim:
            scale_factor = max_dim / float(max(height, width))
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # 2. Optional: Grayscale (Good for simple OCR, maybe skip for LLaVA if color matters)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3. Rename systematically (e.g., class_01.jpg, class_02.jpg)
        new_filename = f"class_{i+1:02d}.jpg"
        save_path = os.path.join(PROCESSED_DIR, new_filename)
        
        cv2.imwrite(save_path, img)
        print(f"Processed: {filename} -> {new_filename}")

if __name__ == "__main__":
    preprocess_images()