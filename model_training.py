from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score



def train_dummy_model(X_train, y_train, X_test, y_test):
    """Train and evaluate a dummy model."""
    to_predict = 1
    dummy = DummyClassifier(strategy="most_frequent", constant=to_predict)
    dummy.fit(X_train, y_train)
    f1_score = dummy.score(X_test, y_test)
    return f1_score

def train_decision_tree(X_train, y_train, X_test, y_test):
    """Train and evaluate a decision tree model."""
    model = DecisionTreeClassifier(criterion="entropy", max_features=1250)
    model.fit(X_train, y_train)
    predictions = model.predict(X_train)
    train_score = accuracy_score(y_train, predictions)
    predictions = model.predict(X_test)
    test_score = accuracy_score(y_test, predictions)
    return train_score, test_score

def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(criterion="entropy", max_features=1250)
    model.fit(X_train, y_train)
    predictions = model.predict(X_train)
    train_score = accuracy_score(y_train, predictions)
    predictions = model.predict(X_test)
    test_score = accuracy_score(y_test, predictions)
    return train_score, test_score