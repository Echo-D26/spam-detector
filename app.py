import re
import streamlit as st
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def clean_text(t):
    t = str(t).lower()
    t = re.sub(r"http\S+|www\S+", " url ", t)
    t = re.sub(r"\d+", " num ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

@st.cache_resource
def learn():
    df = pd.read_csv("SMSSpamCollection", sep="\t", names=["label", "text"])
    df["label"] = df["label"].str.strip().str.lower()
    df = df.dropna().drop_duplicates()
    df["clean"] = df["text"].apply(clean_text)
    df = df[df["clean"] != ""]
    y = (df["label"] == "spam").astype(int)
    model = make_pipeline(TfidfVectorizer(stop_words="english"), MultinomialNB(alpha=0.1))
    model.fit(df["clean"], y)
    return model

model = learn()

st.title("Spam Message Detector")
st.write("Paste a text message below and click Check.")

msg = st.text_area("Your message", height=150)

if st.button("Check"):
    if msg.strip():
        chance = model.predict_proba([clean_text(msg)])[0][1]
        if chance >= 0.5:
            st.error(f"🚨 This looks like SPAM ({chance:.0%} sure)")
        else:
            st.success(f"✅ This looks NOT spam ({1 - chance:.0%} sure)")
    else:
        st.warning("Please paste a message first.")