

# 📄 Resume Clustering and Scoring System

This project analyzes resumes using NLP, clusters them based on job titles, and assigns a relevance score (0 to 10) using semantic matching. It's perfect for learning how AI can help in HR automation.

---

## 🚀 Features

- Resume text preprocessing using NLTK
- Vectorization using TF-IDF
- KMeans clustering based on skills and context
- Similarity scoring using cosine similarity
- Final score between 0 (least relevant) to 10 (most relevant)

---

## 📁 Dataset

You can download the dataset from [Kaggle Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset).

**Expected format** (CSV):
| Job Title | Resume |
|-----------|--------|
| Data Scientist | I have experience in Python, Machine Learning... |

Place your dataset in the `/data` folder as `UpdatedResumeDataSet.csv`.

---


## 🔧 Installation

1. Clone the repo:

```bash
git clone https://github.com/mdzubayerhossain/resume-clustering-scoring.git
cd resume-clustering-scoring
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python src/main.py
```

---

## 📊 Output

- A list of resumes with their **cluster** and **relevance score**
- Bar chart showing number of resumes per cluster

---

## 🧠 Future Improvements

- Add Streamlit UI for recruiters
- Use BERT or Word2Vec embeddings
- Export reports to PDF or Excel

---

## 🤖 Built With

- Python
- NLTK
- Scikit-learn
- Pandas, NumPy
- Matplotlib

---

## 📬 Contact

Made by [Md Zubayer Hossain Patowari](https://github.com/mdzubayerhossain) – feel free to connect!
