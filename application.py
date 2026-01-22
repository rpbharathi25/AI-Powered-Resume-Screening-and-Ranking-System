import streamlit as st
import pandas as pd
from utils import *

st.set_page_config(page_title="AI Resume Matcher", layout="wide")

st.title("🤖 AI Resume Matching System (Industry Grade ATS)")

jd_text = st.text_area("📄 Paste Job Description Here", height=200)

uploaded_files = st.file_uploader(
    "📂 Upload Resume PDFs", type=["pdf"], accept_multiple_files=True
)

if st.button("🚀 Analyze Resumes"):

    if not jd_text or not uploaded_files:
        st.warning("Please provide Job Description and upload resumes.")
        st.stop()

    jd_clean = clean_text(jd_text)
    jd_skills = extract_skills(jd_clean)

    results = []

    for file in uploaded_files:
        resume_text = extract_text_from_pdf(file)
        resume_clean = clean_text(resume_text)

        # --- AI Semantic Similarity ---
        ai_score = semantic_score(jd_clean, resume_clean)

        # --- Skills ---
        resume_skills = extract_skills(resume_clean)
        skill_score, matched_skills, missing_skills = skill_match_score(jd_skills, resume_skills)

        # --- Projects ---
        resume_projects = extract_projects(resume_clean)
        project_score = semantic_score(jd_clean, resume_projects) if resume_projects else 0

        # --- Experience ---
        experience_years = extract_experience(resume_clean)
        exp_score = min(experience_years / 3, 1)  # assuming 3 yrs ideal

        # --- Final Hybrid Score ---
        final_score = (
            0.4 * ai_score +
            0.2 * skill_score +
            0.2 * project_score +
            0.2 * exp_score
        )

        results.append({
            "Resume": file.name,
            "AI Score %": round(ai_score * 100, 2),
            "Skill Match %": round(skill_score * 100, 2),
            "Project Relevance %": round(project_score * 100, 2),
            "Experience (Years)": round(experience_years, 2),
            "Final Match %": round(final_score * 100, 2),
            "Matched Skills": ", ".join(matched_skills),
            "Missing Skills": ", ".join(missing_skills)
        })

    df = pd.DataFrame(results)
    df = df.sort_values(by="Final Match %", ascending=False)

    st.subheader("🏆 Resume Ranking Results")
    st.dataframe(df, use_container_width=True)

    st.subheader("📊 Visual Ranking")

    for _, row in df.iterrows():
        st.write(f"📄 {row['Resume']} — Final Score: {row['Final Match %']}%")
        st.progress(row["Final Match %"] / 100)
