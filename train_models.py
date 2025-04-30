from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import joblib  # to save models

# Train Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
joblib.dump(nb_model, "naive_bayes_model.pkl")

# Train Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
joblib.dump(dt_model, "decision_tree_model.pkl")
