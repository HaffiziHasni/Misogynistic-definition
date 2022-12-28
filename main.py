import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import data_processing
import model_training
import feature_extraction


# Read in and clean the data
df = data_processing.read_csv('data/new.csv')
df = data_processing.clean_data(df)

# Split the data into independent and dependent variables
X, y = data_processing.get_X_and_y(df)

# Extract features from the dependent variable
X = feature_extraction.extract_features(X, y)
y = df['is_misogyny']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train and evaluate the dummy model
f1_score = model_training.train_dummy_model(X_train, y_train, X_test, y_test)
print("f1 score: ", f1_score)

# Train and evaluate the decision tree model
train_score, test_score = model_training.train_decision_tree(X_train, y_train, X_test, y_test)
print("Decision tree train score: ", train_score)
print("Decision tree test score: ", test_score)

# Train and evaluate the random forest model
train_score, test_score = model_training.train_random_forest(X_train, y_train, X_test, y_test)
print("Random forest train score: ", train_score)
print("Random forest test score: ", test_score)

# Read in the new data to be predicted
datanew = pd.read_csv('data/testdata.csv', encoding='latin-1')
only = datanew['Definition']
only = pd.DataFrame(only)

# Extract features from the new data
tfidf_wm = feature_extraction.extract_features(only, only['Definition'])
y = df['is_misogyny'] #idk why, but you need to put this, probably because, initially y is pointed to X.
# Use the random forest model to make predictions on the new data
predicty = model_training.train_random_forest(X_train, y_train, X_test, y_test)
predicty = pd.DataFrame(predicty)
predicty.columns = ['Misogynistic?']

# Combine the new data with the predictions
df_temp = pd.concat([only, predicty], axis=1)
print(df_temp)


"""
text = input("Enter the text you want to check: ")

only = pd.DataFrame({'Definition': [text]})
tfidf_wm = v.transform(only['Definition']).toarray() 
tfidf_wm = pd.DataFrame(tfidf_wm)
predicty = model2.predict(tfidf_wm)
predicty = pd.DataFrame(predicty)

if predicty.iloc[0][0] == 1:
    print("The input text is classified as misogynistic.")
else:
    print("The input text is not classified as misogynistic.")
"""