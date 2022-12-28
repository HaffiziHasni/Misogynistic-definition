from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def extract_features(X, y):
    vectorizer = TfidfVectorizer(use_idf=True)
    X = vectorizer.fit_transform(y)
    tokens = vectorizer.get_feature_names()
    X = pd.DataFrame(data=X.toarray(), index=y, columns=tokens)
    return X
