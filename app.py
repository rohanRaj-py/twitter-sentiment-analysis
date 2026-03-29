import streamlit as st
import pickle
import html
import nltk
import sys
import os

# 🔥 FIX 1: path BEFORE import
sys.path.append(os.path.abspath("."))

# 🔥 FIX 2: silent download
nltk.download('stopwords', quiet=True)

from src.preprocessing import clean_text

# 🔥 FIX 3: safe loading
try:
    model = pickle.load(open('models/model.pkl', 'rb'))
    vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

st.set_page_config(page_title="Sentiment Analysis", layout="wide")

# Title
st.title("🐦 Twitter Sentiment Analyzer")

# Input
text = st.text_area("Write your tweet here:", height=120)

if st.button("Analyze"):

    if text.strip() == "":
        st.warning("⚠️ Enter tweet first")
    else:
        try:
            # Clean text
            cleaned_text = clean_text(text)
            vec = vectorizer.transform([cleaned_text])

            # Prediction
            result = model.predict(vec)[0]
            prob = model.predict_proba(vec)
            confidence = max(prob[0]) * 100

            st.markdown("## 🧾 Tweet Preview")

            safe_text = html.escape(text)

            st.markdown(f"""
            <div style="
                background-color:#1e1e2f;
                padding:20px;
                border-radius:15px;
                display:flex;
                align-items:flex-start;
                gap:15px;
            ">
                <img src="https://cdn-icons-png.flaticon.com/512/149/149071.png"
                     width="50" height="50" style="border-radius:50%;"/>

                <div>
                    <b style="color:white;">@user</b><br>
                    <span style="color:#ccc;">{safe_text}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("## 📊 Result")

            if result == 1:
                st.markdown(f"""
                <div style="
                    background-color:#0f5132;
                    padding:20px;
                    border-radius:12px;
                    text-align:center;
                    font-size:22px;
                    color:#d1e7dd;">
                    😊 Positive Sentiment <br>
                    Confidence: {confidence:.2f}%
                </div>
                """, unsafe_allow_html=True)

                st.balloons()
            else:
                st.markdown(f"""
                <div style="
                    background-color:#842029;
                    padding:20px;
                    border-radius:12px;
                    text-align:center;
                    font-size:22px;
                    color:#f8d7da;">
                    😠 Negative Sentiment <br>
                    Confidence: {confidence:.2f}%
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error during prediction: {e}")