import pandas as pd
import numpy as np
import nltk
import re
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load dataset
data = pd.read_csv('./data/resume_dataset.csv')  # Make sure this path is correct
print("✅ Dataset Loaded")

# Clean and preprocess resume text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))  # Remove non-alphabetic characters
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Apply cleaning
data['Cleaned_Resume'] = data['Resume'].apply(clean_text)

# Vectorization using TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(data['Cleaned_Resume'])

# KMeans clustering
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)
print("✅ Clustering Complete")

# Function to calculate score (0–10) based on job title
def get_similarity_score(resume_text, job_title):
    resume_vec = vectorizer.transform([clean_text(resume_text)])
    title_vec = vectorizer.transform([clean_text(job_title)])
    score = cosine_similarity(resume_vec, title_vec)[0][0]
    return round(score * 10, 2)

# Score each resume
data['Score'] = data.apply(lambda row: get_similarity_score(row['Resume'], row['Job Title']), axis=1)

# Output results
print("\n🎯 Sample Scored Data:")
print(data[['Job Title', 'Score', 'Cluster']].head())

# Visualize number of resumes per cluster
plt.figure(figsize=(6, 4))
data['Cluster'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title("Number of Resumes per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Resumes")
plt.tight_layout()
plt.savefig("resume_clusters.png")
plt.show()

# Optionally save the output
data.to_csv('./data/resume_scored_output.csv', index=False)
print("\n📄 Results saved to data/resume_scored_output.csv")
