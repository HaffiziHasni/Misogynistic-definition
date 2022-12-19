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




df=pd.read_csv('new.csv',encoding='latin-1')
#cleaning stop words and put into new column
df['cleaned_definition_nostop'] = df['cleaned_definition'].apply(lambda x: ' '.join([item for item in x.split() if item not in STOPWORDS]))

y=df['cleaned_definition_nostop']
#need to put into a variable to be transformed
v= TfidfVectorizer(use_idf=True)
x=v.fit_transform(y)
#getting feature names
tokens=v.get_feature_names()
#TD-IF vectorizer
df_s=pd.DataFrame(data=x.toarray(),index=y, columns=tokens)
#creating variables where independentvar is the x and dependent var is the y
independentVar=df_s
dependentVar=df['is_misogyny']
#move directly to model training, if we were to convert tfidf to csv now, it'll take long 
#splitting data into training and testing set
x_train,x_test,y_train,y_test=train_test_split(independentVar,dependentVar, test_size=0.2)

#training the dummyclassifier model
toPredict=1
dummy=DummyClassifier(strategy="most_frequent", constant=toPredict)
dummy.fit(x_train,y_train)
f1_score=dummy.score(x_test,y_test)
print("f1 score: ",f1_score)

model=DecisionTreeClassifier(criterion="entropy", max_features=1250) #select algo, can change, select the best one
model.fit(x_train, y_train) #fitting model to data
#checking model performance on training data
predictions = model.predict(x_train)
print( "Train score: ",accuracy_score(y_train, predictions))
# Evaluate the model on  test data
predictions = model.predict(x_test)
predictions
print("Test score: ",accuracy_score(y_test, predictions))

model2=RandomForestClassifier(criterion="entropy", max_features=1250) #select algo, can change, select the best one
model2.fit(x_train, y_train) #fitting model to data
#checking model performance on training data
predictions = model2.predict(x_train)
print(accuracy_score(y_train, predictions))
predictions = model2.predict(x_test)
print(accuracy_score(y_test, predictions))

datanew=pd.read_csv('testdata.csv',encoding='latin-1')

#datanew['Definition'] = datanew['Definition'].apply(lambda x: ' '.join([item for item in x.split() if item not in STOPWORDS]))

only=datanew['Definition']
only=pd.DataFrame(only)

print(type(only))

#always do this!
tfidf_wm = v.transform(only['Definition']).toarray() 
tfidf_wm=pd.DataFrame(tfidf_wm)

predicty=model2.predict(tfidf_wm)

predicty=pd.DataFrame(predicty)
predicty.columns=['Misogynistic?']



df_temp=pd.concat([only,predicty],axis=1)
print(df_temp) 

#ask input from user and whether it is misogynistic
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