import pandas as pd
from wordcloud import STOPWORDS

def read_csv(filepath):
    df = pd.read_csv(filepath, encoding='latin-1')
    return df

def clean_data(df):
    df['cleaned_definition_nostop'] = df['cleaned_definition'].apply(lambda x: ' '.join([item for item in x.split() if item not in STOPWORDS]))
    return df

def get_X_and_y(df):
    y = df['cleaned_definition_nostop']
    X = df.drop(columns=['cleaned_definition_nostop', 'is_misogyny'])
    return X, y