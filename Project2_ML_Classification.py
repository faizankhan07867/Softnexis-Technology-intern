# PROJECT 2: Machine Learning Classification
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42)

dt=DecisionTreeClassifier(random_state=42)
dt.fit(X_train,y_train)

y_pred=dt.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

plt.figure(figsize=(12,8))
plot_tree(dt, feature_names=iris.feature_names,
          class_names=iris.target_names, filled=True)
plt.title("Decision Tree")
plt.show()

rf=RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train,y_train)

rf_pred=rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test,rf_pred))
print(classification_report(y_test,rf_pred))

print("Project 2 Completed Successfully!")
