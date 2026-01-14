# 🧠 AI Resume Role Predictor

An AI-powered application that predicts the most suitable job role based on resume content or manually entered skills. It uses machine learning (SVM + TF-IDF) to classify resumes across multiple roles and provides a simple, interactive Streamlit UI.

🚀 Project Overview

Most resumes are screened automatically by Applicant Tracking Systems (ATS). This project simulates that behavior by:

Extracting relevant text from resumes (PDF)

Cleaning & processing the content

Predicting the most suitable job role using ML

Allowing manual skill input for quick predictions

# ✨ Features

✔ Predicts job role based on skills or resume text
✔ Supports PDF resume uploads (digital PDFs)
✔ Manual skill input mode
✔ TF-IDF feature extraction
✔ SVM-based classification model
✔ Clean and user-friendly Streamlit UI

# 🏗 Tech Stack
Layer	Tools Used
Frontend/UI	Streamlit
Machine Learning	Scikit-Learn (SVM, TF-IDF)
Language Processing	Regex, TF-IDF
Resume Parsing	pdfplumber
Model Persistence	Pickle
Deployment	Streamlit Cloud / Local
📦 Project Structure
📁 Job-Role-Predictor
 ├── app.py
 ├── svm_model.pkl
 ├── vectorizer.pkl
 ├── label_encoder.pkl
 ├── requirements.txt
 ├── README.md

# 🎯 How it Works

Input Options

Upload PDF resume (digital only)

OR enter skills manually

Text Extraction

Uses pdfplumber for digital resumes

Text Cleaning

Lowercasing

Removing punctuation/special chars

Removing extra spaces

Vectorization

TF-IDF Vectorizer converts text → numeric vectors

Classification

SVM model predicts the most suitable job role

Output

Shows predicted role in UI

# 🧪 Model Details

Algorithm: Support Vector Machine (LinearSVC)

Feature Extraction: TF-IDF

Classes: 60+ job roles (ex: Data Analyst, Backend Developer, AI Engineer, etc.)

Accuracy: ~96% (varies by dataset)

# 🛠 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/Job-Role-Predictor.git
cd Job-Role-Predictor

2️⃣ Create Virtual Environment (Optional)
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run Application
streamlit run app.py


App will open at:

http://localhost:8501

# 📝 Usage
Manual Skill Input Example:

Python, TensorFlow, Machine Learning, SQL

Output:

Predicted Role: Machine Learning Engineer

Resume Upload

Upload any digital PDF resume and click Predict.

# 🌐 Deployment

Compatible with:

Streamlit Cloud (recommended for no OCR)

Render

HuggingFace Spaces (with OCR enabled)

# 📌 Future Enhancements

OCR support for scanned PDFs & images

Resume scoring (ATS-style percentage)

Job recommendations

Missing skill suggestions

Profile matching against JD

Semantic embeddings (BERT, SBERT)

NER-based skill extraction

