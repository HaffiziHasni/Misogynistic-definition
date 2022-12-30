from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def extract_features(df,y):
    """Extract features from the data using a TfidfVectorizer."""
    descriptions = df['cleaned_definition'].apply(str).apply(lambda x: x.lower())
    vectorizer = TfidfVectorizer(use_idf=True)
    X = vectorizer.fit_transform(descriptions)
    feature_names = vectorizer.get_feature_names()
    df_sparse = pd.DataFrame(data=X.toarray(), index=descriptions, columns=feature_names)
    return df_sparse
