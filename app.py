import streamlit as st
import joblib
import numpy as np
import re

# ✅ Move page config to very top
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f7f9fc;
        }
        .title {
            font-size: 40px;
            font-weight: 700;
            text-align: center;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #7f8c8d;
            margin-bottom: 30px;
        }
        .stTextArea textarea {
            background-color: #ffffff;
            border-radius: 10px;
            border: 1px solid #dcdde1;
        }
        .result {
            font-size: 28px;
            font-weight: bold;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\n', ' ', text)
    return text

# App Title
st.markdown('<p class="title">📰 Fake News Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter a news article below and find out if it is Real or Fake</p>', unsafe_allow_html=True)

# Text input area
news_input = st.text_area("✍️ Paste the News Article:", height=200)

# Predict button
if st.button("🔍 Analyze News"):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter a news article to analyze.")
    else:
        cleaned = clean_text(news_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        prob = model.predict_proba(vectorized)[0]

        if prediction == 0:
            st.success("✅ This looks like **REAL** news!")
        else:
            st.error("❌ This appears to be **FAKE** news!")

        st.write(f"**📊 Probability of Real:** {prob[0]:.4f}")
        st.write(f"**📉 Probability of Fake:** {prob[1]:.4f}")

# Footer
st.markdown("""
    <hr>
    <p style="text-align: center; color: grey;">
        Made with ❤️ using Streamlit | Fake News Detector
    </p>
""", unsafe_allow_html=True)
