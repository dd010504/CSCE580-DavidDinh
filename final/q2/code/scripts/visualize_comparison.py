import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
DATA_DIR = "./"  # Assumes you run this from the 'scripts' folder where CSVs are saved
MANUAL_FILE = "final_report_table.csv"
OCR_FILE = "ai_ocr_results.csv"
LLAVA_FILE = "llava_results.csv"
OUTPUT_IMG = "model_performance_comparison.png"

def main():
    print("--- Generating Comparison Graphs ---")
    
    # 1. Load DataFrames
    # We use 'Class' as the index to make merging easy
    try:
        df_manual = pd.read_csv(MANUAL_FILE).set_index('Class')
        df_manual = df_manual[['Count']].rename(columns={'Count': 'Manual (Truth)'})
    except:
        print(f"Could not load {MANUAL_FILE}. Run analyze_attendance.py first.")
        return

    dfs_to_merge = [df_manual]

    if os.path.exists(OCR_FILE):
        df_ocr = pd.read_csv(OCR_FILE).set_index('Class')
        # We want the 'AI_Matched_Count' column
        df_ocr = df_ocr[['AI_Matched_Count']].rename(columns={'AI_Matched_Count': 'EasyOCR'})
        dfs_to_merge.append(df_ocr)
    
    if os.path.exists(LLAVA_FILE):
        df_llava = pd.read_csv(LLAVA_FILE).set_index('Class')
        # We want the 'Matched_Count' column
        df_llava = df_llava[['Matched_Count']].rename(columns={'Matched_Count': 'LLaVA'})
        dfs_to_merge.append(df_llava)

    # 2. Merge All Data
    # Inner join matches only classes present in all files, Outer joins everything
    df_final = pd.concat(dfs_to_merge, axis=1, join='outer').fillna(0)
    
    # Sort by Class Number (extract digits)
    df_final['SortNum'] = df_final.index.str.extract(r'(\d+)').astype(float)
    df_final = df_final.sort_values('SortNum').drop(columns=['SortNum'])

    print("Combined Data Sample:")
    print(df_final.head())

    # 3. Plotting with Pandas
    plt.figure(figsize=(14, 7))
    
    # Plot bar chart directly from Pandas
    # width=0.8 makes bars grouped nicely
    ax = df_final.plot(kind='bar', width=0.8, figsize=(14, 7), 
                       color=['#2ca02c', '#1f77b4', '#ff7f0e']) # Green (Manual), Blue (OCR), Orange (LLaVA)

    plt.title("Attendance Audit: Human vs. AI Performance", fontsize=16)
    plt.ylabel("Students Identified", fontsize=12)
    plt.xlabel("Class Session", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Method")

    # Add text annotation for the average
    avg_manual = df_final['Manual (Truth)'].mean()
    avg_ocr = df_final['EasyOCR'].mean() if 'EasyOCR' in df_final.columns else 0
    avg_llava = df_final['LLaVA'].mean() if 'LLaVA' in df_final.columns else 0

    stats_text = (f"Avg Accuracy vs Truth:\n"
                  f"EasyOCR: {(avg_ocr/avg_manual)*100:.1f}%\n"
                  f"LLaVA: {(avg_llava/avg_manual)*100:.1f}%")
    
    plt.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    
    # 4. Save
    plt.savefig(OUTPUT_IMG)
    print(f"\n[SUCCESS] Graph saved to {OUTPUT_IMG}")

if __name__ == "__main__":
    main()