# 🧠 Resume Relevance Checker — Streamlit App

This is a smart, interactive Streamlit web app that evaluates how well a candidate's resume matches a given job description using natural language processing (NLP). It provides a relevance score and a fit verdict—without relying on external APIs.

## 🚀 Features

- Upload resume files in `.pdf`, `.docx`, or `.txt` format
- Paste job description text directly into the app
- Calculates relevance score using:
  - 🔍 TF-IDF keyword matching
  - 🧠 Semantic similarity via spaCy
  - 📊 Cosine similarity for scoring
- Generates a final score (0–100) and a fit verdict:
  - ✅ High Fit (75–100)
  - ⚖️ Medium Fit (50–74)
  - ❌ Low Fit (0–49)
- Clean and responsive Streamlit interface
- Modular codebase for easy extension

## 🛠 Tech Stack

| Component        | Technology Used            |
|------------------|----------------------------|
| Frontend         | Streamlit                  |
| Backend Logic    | Python                     |
| NLP              | spaCy, scikit-learn        |
| File Parsing     | PyPDF2, python-docx        |
| Text Processing  | re, string, io             |
| Similarity Scoring | TF-IDF, cosine similarity |
| Deployment Ready | Streamlit Cloud compatible |

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Kambhampati-Sruthi/Resume-Checker.git
   cd Resume-Checker
