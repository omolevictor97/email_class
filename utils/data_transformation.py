import pandas as pd
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
from constants import CUSTOM_STOPWORDS
porter_stemmer = PorterStemmer()


def data_transformer(df):
    corpus = []

    for i in range(0, len(df)):
        review = re.sub('[^a-zA-Z0-9]', " ", df['Messages'][i]) # remove all non alphabetic symbols
        review = review.lower() #convert all letters to lower alphabetics
        review = review.split() # split words to return a list of words
        review = [porter_stemmer.stem(word) for word in review if not word in CUSTOM_STOPWORDS] # returns all root words
        review = " ".join(review)
        corpus.append(review)
    return corpus
