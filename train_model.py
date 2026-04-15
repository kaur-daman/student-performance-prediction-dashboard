import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

data = pd.read_csv("data/clean_dataset.csv")

X = data.drop("performance",axis=1)
y = data["performance"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

models = {
    "RandomForest":RandomForestClassifier(n_estimators=200),
    "DecisionTree":DecisionTreeClassifier(),
    "SVM":SVC()
}

scores = {}

for name,model in models.items():
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test,pred)
    scores[name] = acc

print(scores)

best_model = models[max(scores,key=scores.get)]

pickle.dump(best_model,open("models/student_model.pkl","wb"))
