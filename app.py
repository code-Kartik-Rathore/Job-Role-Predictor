# import streamlit as st
# import pickle
# import re
# import pdfplumber
# from pdf2image import convert_from_path
# from PIL import Image
# import pytesseract
# import numpy as np
# import cv2

# # ================== PAGE CONFIG ==================
# st.set_page_config(
#     page_title="AI Resume Role Predictor",
#     page_icon="💼",
#     layout="wide"
# )

# # ================== LOAD ML COMPONENTS ==================
# @st.cache_resource
# def load_components():
#     model = pickle.load(open("svm_model.pkl", "rb"))
#     vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
#     le = pickle.load(open("label_encoder.pkl", "rb"))
#     return model, vectorizer, le

# model, vectorizer, le = load_components()

# # ================== TEXT CLEANING ==================
# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r'[^a-zA-Z\s]', ' ', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# # ================== PREDICTION FUNCTION ==================
# def predict_role(text):
#     cleaned = clean_text(text)
#     vec = vectorizer.transform([cleaned])
#     pred = model.predict(vec)[0]
#     return le.inverse_transform([pred])[0]
# # ================== TEXT VALIDATION FUNCTION ==================
# def validate_resume_text(text):
#     text = text.lower()
    
#     # 1. Empty / too short
#     if len(text.strip()) < 100:
#         return "too_empty"
    
#     # 2. Missing skill keywords
#     SKILL_KEYWORDS = [
#         "python","java","react","node","sql","aws","git","docker","html","css",
#         "javascript","machine learning","data","nlp","tensorflow","pytorch",
#         "c++","api","linux","cloud","devops","analysis","backend","frontend"
#     ]
#     if not any(skill in text for skill in SKILL_KEYWORDS):
#         return "no_skills"
    
#     # 3. Off-context detection
#     OFF_CONTEXT_WORDS = [
#         "poem","poetry","novel","chapter","verse","wedding","invoice","bill",
#         "receipt","ticket","train","recipe","food","hotel","lyrics"
#     ]
#     off_hits = sum(word in text for word in OFF_CONTEXT_WORDS)
#     if off_hits > 3:
#         return "off_context"
    
#     return "valid"


# # ================== PDF TEXT EXTRACTION ==================
# def extract_text_from_pdf(pdf_file):
#     text = ""
#     with pdfplumber.open(pdf_file) as pdf:
#         for page in pdf.pages:
#             page_text = page.extract_text()
#             if page_text:
#                 text += page_text + " "
#     return text

# # ================== OCR FOR SCANNED PDF ==================
# def extract_text_from_scanned_pdf(pdf_file):
#     text = extract_text_from_pdf(pdf_file)

#     if len(text.strip()) < 50:  # fallback to OCR
#         st.warning("Scanned PDF detected — using OCR...")
#         images = convert_from_path(pdf_file)
#         for img in images:
#             img_np = np.array(img)
#             img_cv = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
#             text += pytesseract.image_to_string(img_cv)
#     return text

# # ================== OCR FOR IMAGE ==================
# def ocr_image(image_file):
#     image = Image.open(image_file)
#     img_np = np.array(image)
#     img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
#     text = pytesseract.image_to_string(img)
#     return text

# # ================== UI HEADER ==================
# st.title("💼 AI-Powered Resume Role Predictor")
# st.write("Upload your resume or skills to find your best matching job role.")

# # ================== INPUT OPTIONS ==================
# option = st.radio(
#     "Choose Input Method:",
#     ("Add Skills Manually", "Upload Resume PDF", "Upload Resume Image (jpg/png)")
# )

# # ================== MODE A: MANUAL SKILLS ==================
# if option == "Add Skills Manually":
#     skills = st.text_area("Enter skills (comma-separated):",
#                           placeholder="Example: Python, TensorFlow, SQL, Machine Learning")
#     if st.button("Predict Role"):
#         if skills.strip() == "":
#             st.error("Please enter some skills.")
#         else:
#             predicted = predict_role(skills)
#             st.success(f"🎯 Predicted Job Role: **{predicted}**")

# # ================== MODE B: PDF UPLOAD ==================
# elif option == "Upload Resume PDF":
#     uploaded_pdf = st.file_uploader("Upload PDF File", type=["pdf"])
    
#     if uploaded_pdf is not None:
#         with open("temp_resume.pdf", "wb") as f:
#             f.write(uploaded_pdf.read())
        
#         extracted_text = extract_text_from_scanned_pdf("temp_resume.pdf")
        
#         #validate
#         status = validate_resume_text(extracted_text)

#         if status == "too_empty":
#             st.error("❗ Resume content is too short or unreadable.")
#             st.stop()
#         elif status == "no_skills":
#             st.error("❗ No relevant technical skills detected in the document.")
#             st.stop()
#         elif status == "off_context":
#             st.error("❗ Uploaded document appears unrelated to resumes or job skills.")
#             st.stop()

#         st.subheader("📄 Extracted Resume Text Preview:")
#         st.write(extracted_text[:1500] + "...")
        
#         if st.button("Predict Role from PDF"):
#             predicted = predict_role(extracted_text)
#             st.success(f"🎯 Predicted Job Role: **{predicted}**")

# # ================== MODE C: IMAGE UPLOAD ==================
# elif option == "Upload Resume Image (jpg/png)":
#     uploaded_image = st.file_uploader("Upload Image File", type=["jpg", "jpeg", "png"])
    
#     if uploaded_image is not None:
#         st.image(uploaded_image, caption="Uploaded Resume Image", use_column_width=True)
        
#         extracted_text = ocr_image(uploaded_image)
#         status = validate_resume_text(extracted_text)

#         if status == "too_empty":
#             st.error("❗ Resume content is too short or unreadable.")
#             st.stop()
#         elif status == "no_skills":
#             st.error("❗ No relevant technical skills detected in the document.")
#             st.stop()
#         elif status == "off_context":
#             st.error("❗ Uploaded document appears unrelated to resumes or job skills.")
#             st.stop()
        
#         st.subheader("🖼 OCR Extracted Text Preview:")
#         st.write(extracted_text[:1500] + "...")
        
#         if st.button("Predict Role from Image"):
#             predicted = predict_role(extracted_text)
#             st.success(f"🎯 Predicted Job Role: **{predicted}**")

# # ================== FOOTER ==================
# st.markdown("---")
# st.markdown("<center>Made by Kartik Rathore</center>", unsafe_allow_html=True)

import streamlit as st
import pickle
import re
import pdfplumber

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="AI Resume Role Predictor", page_icon="💼", layout="wide")

# ================== LOAD ML COMPONENTS ==================
@st.cache_resource
def load_components():
    model = pickle.load(open("svm_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    le = pickle.load(open("label_encoder.pkl", "rb"))
    return model, vectorizer, le

model, vectorizer, le = load_components()

# ================== TEXT CLEANING ==================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

# ================== PREDICTION FUNCTION ==================
def predict_role(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    return le.inverse_transform([pred])[0]

# ================== PDF TEXT EXTRACTION ==================
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text

# ================== UI SECTION ==================
st.title("💼 AI-Powered Resume Role Predictor")
st.write("Upload resume or manually enter skills to get your best matching job role.")

option = st.radio("Choose input method:", ("Add Skills Manually", "Upload Resume PDF"))

# ---- Manual Skills ----
if option == "Add Skills Manually":
    skills = st.text_area("Enter skills (comma-separated):", placeholder="Python, SQL, React, Machine Learning")
    if st.button("Predict"):
        if not skills.strip():
            st.error("Please enter some skills to continue.")
        else:
            role = predict_role(skills)
            st.success(f"🎯 Predicted Job Role: **{role}**")

# ---- PDF Upload ----
else:
    uploaded_pdf = st.file_uploader("Upload PDF Resume", type=["pdf"])
    if uploaded_pdf:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_pdf.read())
        
        text = extract_text_from_pdf("temp.pdf")
        
        if len(text.strip()) < 100:
            st.error("❗ Resume content is too short or not readable.")
        else:
            st.subheader("📄 Resume Text Preview:")
            st.write(text[:1000] + " ...")
            
            if st.button("Predict from PDF"):
                role = predict_role(text)
                st.success(f"🎯 Predicted Job Role: **{role}**")

st.markdown("---")
st.markdown("<center>Made by Kartik Rathore </center>", unsafe_allow_html=True)
