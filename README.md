# 📰 Fake News Detection using Machine Learning

A Machine Learning-based web app that detects whether a news article is **Real or Fake** using **Logistic Regression** and **TF-IDF Vectorization**. The model is deployed using **Streamlit** for easy interaction.

---

##  Features

-  Detects Fake vs Real news with high accuracy  
-  Machine Learning Model — Logistic Regression  
-  Text preprocessing (removing punctuation, URLs, symbols, etc.)  
-  Streamlit web interface for real-time predictions  
-  Model saved using `joblib` (`model.pkl`, `vectorizer.pkl`)  

---

## 🛠️ Technologies Used

| Tool/Library     | Purpose |
|------------------|---------|
| Python           | Main Programming Language |
| Pandas, NumPy    | Data Handling |
| Scikit-learn     | TF-IDF & Logistic Regression |
| re (Regex)       | Text Cleaning |
| Streamlit        | Web Application |
| Joblib           | Model Saving/Loading |

---

##  Files in the Project

- `app.py` → Streamlit web app  
- `model.pkl` → Trained Logistic Regression model  
- `vectorizer.pkl` → TF-IDF vectorizer  
- `True.csv`, `Fake.csv` → Dataset used for training  
- `README.md` → Project documentation  

---

##  How to Run the Project

```bash
# 1. Install dependencies
pip install streamlit pandas numpy scikit-learn joblib

# 2. Run the Streamlit app
streamlit run app.py
