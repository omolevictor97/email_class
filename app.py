import streamlit as st
import pandas as pd
import re
import contractions
import string
from nltk.stem.porter import PorterStemmer
import pickle
from constants import CUSTOM_STOPWORDS


ps = PorterStemmer()
def preprocess(text):
    corpus = []
    text = re.sub('[^a-zA-Z0-9]', " ", text)
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    text = url_pattern.sub(r"", text)
    text = text.lower()
    text = contractions.fix(text, slang=True)
    text = string.punctuation
    text = text.translate(str.maketrans("", "", text))
    text = text.split()
    text = [ps.stem(word) for word in text if not word in CUSTOM_STOPWORDS]
    text = " ".join(text)
    corpus.append(text)
    return corpus


st.title("Email Classificaton")
user_input = st.text_input("Enter your email message: ")

def main():
    corpus = [user_input]
    with open("tf_idf.pkl", "rb") as cv_files:
        loaded_cv = pickle.load(cv_files)
        transformed_input = loaded_cv.transform(corpus)
        print(transformed_input)

    with open("model.pkl", "rb") as log_reg_model:
        model = pickle.load(log_reg_model)
        predicted = model.predict(transformed_input.toarray())
        if predicted[0] == 0:
            predicted = "Ham"
        else:
            predicted = "Spam"
        print(predicted)
    st.success(f"Your message is of type {predicted}")

if st.button("Predict"):
     main()
     st.balloons()
