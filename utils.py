import fitz
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

SKILL_LIST = [
    "python","java","c","c++","sql","html","css","javascript",
    "machine learning","deep learning","data science","nlp",
    "tensorflow","pytorch","docker","aws","spark","react","flask","django"
]

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text.lower()

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    return text.lower()

def extract_skills(text):
    found = []
    for skill in SKILL_LIST:
        if skill in text:
            found.append(skill)
    return list(set(found))

def extract_experience(text):
    matches = re.findall(r'(\d+)\s+(years|year|yrs|months|month)', text)
    total_years = 0
    for val, unit in matches:
        val = int(val)
        if "month" in unit:
            total_years += val / 12
        else:
            total_years += val
    return total_years

def extract_projects(text):
    keywords = ["project", "projects", "experience", "work"]
    sentences = text.split(".")
    proj_lines = [s for s in sentences if any(k in s for k in keywords)]
    return " ".join(proj_lines)

def semantic_score(text1, text2):
    emb1 = model.encode([text1])
    emb2 = model.encode([text2])
    return float(cosine_similarity(emb1, emb2)[0][0])

def skill_match_score(jd_skills, resume_skills):
    if not jd_skills:
        return 0
    matched = set(jd_skills) & set(resume_skills)
    return len(matched) / len(jd_skills), list(matched), list(set(jd_skills) - set(resume_skills))
