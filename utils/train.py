from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_recall_curve, precision_score
import pickle


def split(corpus, y):
    X_train, X_test, y_train, y_test = train_test_split(corpus, y, test_size=0.2, shuffle=True, stratify=y, random_state=42)
    y_train = y_train.map({"spam": 1, "ham" : 0})

    y_test = y_test.map({"spam": 1, "ham" : 0})
    y_train.astype(int)
    y_test.astype(int)

    return X_train, X_test, y_train, y_test


def train(X_train, X_test, y_train, y_test):
    tf_idf = TfidfVectorizer(max_features=2000, binary=True, ngram_range=(1,2))
    NB_model = GaussianNB()
    X_train = tf_idf.fit_transform(X_train).toarray()
    X_test = tf_idf.transform(X_test).toarray()

    print(X_train.shape)
    print(X_test.shape)
    NB_model.fit(X_train, y_train)
    train_score = NB_model.score(X_train, y_train)
    test_score = NB_model.score(X_test, y_test)


    print(f"Train Score: {train_score} \n Test Score: {test_score}")

    return tf_idf, NB_model

def save_objs(model, tf_idf):
    
    with open("tf_idf.pkl", "wb") as cv_files:
        pickle.dump(tf_idf, cv_files)
    
    with open("model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)
