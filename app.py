import streamlit as st
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

@st.cache_resource
def learn():
    df = pd.read_csv("SMSSpamCollection", sep="\t", names=["label","text"])
    m = make_pipeline(TfidfVectorizer(stop_words="english"), MultinomialNB())
    m.fit(df.text, df.label)
    return m

    model = learn()
    st.title("Spam Message Detector")
    msg = st.text_area("Paste a message here")

    if st.button("Check"):
        if msg.strip():
            if model.predict([msg])[0] == "spam":
                st.error("🚨 This looks like SPAM")
            else:
                st.success("✅ This looks NOT spam")