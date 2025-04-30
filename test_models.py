import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load test data
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv').values.ravel()

# Load models
nb_model = joblib.load("data/naive_bayes_model.pkl")
dt_model = joblib.load("data/decision_tree_model.pkl")
lr_model = joblib.load("data/logistic_regression_model.pkl")
rf_model = joblib.load("data/random_forest_model.pkl")

# Evaluation function
def evaluate(y_true, y_pred, model_name):
    print(f"\n--- {model_name} Evaluation ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# Predict and evaluate each model
evaluate(y_test, nb_model.predict(X_test), "Naive Bayes")
evaluate(y_test, dt_model.predict(X_test), "Decision Tree")
evaluate(y_test, lr_model.predict(X_test), "Logistic Regression")
evaluate(y_test, rf_model.predict(X_test), "Random Forest")