import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from rank_bm25 import BM25Okapi  # type: ignore
from fuzzywuzzy import fuzz  # type: ignore

# Normalize text: lowercase, remove punctuation
def normalize_text(text_list):
    normalized_list = []
    for text in text_list:
        if isinstance(text, str):
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
            normalized_list.append(text)
        else:
            normalized_list.append("")
    return normalized_list

# TF-IDF cosine similarity
def calculate_tfidf_similarity(resume_items, jd_items):
    if not resume_items or not jd_items:
        return 0.0

    all_items = resume_items + jd_items
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_items)

    resume_vector = tfidf_matrix[:len(resume_items)].sum(axis=0)
    jd_vector = tfidf_matrix[len(resume_items):].sum(axis=0)

    # Convert np.matrix to np.array
    resume_vector = np.asarray(resume_vector)
    jd_vector = np.asarray(jd_vector)

    similarity = cosine_similarity(resume_vector, jd_vector)[0][0]
    return similarity * 100

# BM25 relevance scoring
def calculate_bm25_score(resume_items, jd_items):
    if not resume_items or not jd_items:
        return 0.0

    tokenized_corpus = [doc.split(" ") for doc in jd_items]
    bm25 = BM25Okapi(tokenized_corpus)

    scores = []
    for resume_item in resume_items:
        query = resume_item.split(" ")
        doc_scores = bm25.get_scores(query)
        scores.append(max(doc_scores) if doc_scores.size > 0 else 0)

    max_possible_score = len(resume_items) * 10
    total_score = sum(scores)
    percentage = (total_score / max_possible_score) * 100 if max_possible_score > 0 else 0.0
    return min(percentage, 100.0)

# Fuzzy string matching
def calculate_fuzzy_match(resume_items, jd_items, threshold=80):
    if not resume_items or not jd_items:
        return 0.0

    matched_count = 0
    for r_item in resume_items:
        for jd_item in jd_items:
            if fuzz.ratio(r_item.lower(), jd_item.lower()) >= threshold:
                matched_count += 1
                break

    percentage = (matched_count / len(resume_items)) * 100
    return percentage

# Main matching function
def match_resume_to_jd(parsed_resume, parsed_jd):
    resume_skills = normalize_text(parsed_resume.get("Skills", []))
    jd_must_have_skills = normalize_text(parsed_jd.get("MustHaveSkills", []))
    jd_good_to_have_skills = normalize_text(parsed_jd.get("GoodToHaveSkills", []))
    resume_education = normalize_text(parsed_resume.get("Education", []))
    jd_qualifications = normalize_text(parsed_jd.get("RequiredQualifications", []))
    resume_experience = normalize_text([
        exp for exp_list in parsed_resume.get("Experience", [])
        for exp in exp_list.split('\n') if exp.strip()
    ])

    all_jd_skills = jd_must_have_skills + jd_good_to_have_skills
    tfidf_skill_score = calculate_tfidf_similarity(resume_skills, all_jd_skills)
    bm25_skill_score = calculate_bm25_score(resume_skills, all_jd_skills)
    fuzzy_must_have_skill_score = calculate_fuzzy_match(resume_skills, jd_must_have_skills)

    fuzzy_education_score = calculate_fuzzy_match(resume_education, jd_qualifications)
    tfidf_experience_score = calculate_tfidf_similarity(resume_experience, jd_qualifications)
    bm25_experience_score = calculate_bm25_score(resume_experience, jd_qualifications)

    total_weight = 0
    weighted_sum = 0

    weighted_sum += fuzzy_must_have_skill_score * 0.40
    total_weight += 0.40

    weighted_sum += (tfidf_skill_score * 0.5 + bm25_skill_score * 0.5) * 0.30
    total_weight += 0.30

    weighted_sum += fuzzy_education_score * 0.15
    total_weight += 0.15

    weighted_sum += (tfidf_experience_score * 0.5 + bm25_experience_score * 0.5) * 0.15
    total_weight += 0.15

    hard_match_score = int(weighted_sum / total_weight) if total_weight > 0 else 0
    return min(hard_match_score, 100)

# Test data
if __name__ == '__main__':
    dummy_resume = {
        "Name": "John Doe",
        "Education": ["University of ABC - Bachelor of Science in Computer Science"],
        "Skills": ["Python", "Flask", "NLP", "Machine Learning", "TensorFlow", "Keras", "Data Analysis"],
        "Projects": [],
        "Certifications": [],
        "Experience": [
            "Software Engineer - Company X (2020-Present)",
            "Developed and maintained backend services using Python and Flask.",
            "Implemented machine learning models for data prediction."
        ],
        "RawContent": "..."
    }

    dummy_jd = {
        "RoleTitle": "Software Engineer",
        "MustHaveSkills": ["Python", "Flask", "NLP", "Machine Learning"],
        "GoodToHaveSkills": ["TensorFlow", "AWS"],
        "RequiredQualifications": [
            "Bachelor's degree in Computer Science or related field",
            "3+ years of experience in backend development",
            "Proficiency in Python and web frameworks"
        ],
        "RawContent": "..."
    }

    match_score = match_resume_to_jd(dummy_resume, dummy_jd)
    print(f"Hard Match Score: {match_score}%")
