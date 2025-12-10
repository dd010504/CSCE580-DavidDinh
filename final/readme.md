David Dinh CSCE 580
# Attendance Audit System - Setup & Execution Guide

This project contains the complete pipeline to ingest raw attendance sheets, preprocess them, and analyze student attendance trends using both manual ground-truth verification and experimental AI models (EasyOCR, LLaVA).

---

## 1. Prerequisites & Dependencies

### System Requirements

- **Python 3.8+**: Ensure Python is installed and added to your system path.
- **Ollama (Optional)**: Required only if you intend to run the Generative AI (LLaVA) pipeline.
  - Download from [ollama.com](https://ollama.com).
  - Open your terminal and run:
    ```bash
    ollama pull llava
    ```

### Python Libraries

Install the required dependencies by running the following command in your terminal:

```bash
pip install pandas opencv-python matplotlib easyocr fuzzywuzzy python-Levenshtein ollama
```


answers for q1 and q2 are in their respectable folder.