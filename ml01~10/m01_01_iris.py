from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, load_digits, load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load all scikit-learn datasets
datasets = [
    ("Iris", load_iris()),
    ("Breast Cancer", load_breast_cancer()),
    ("Diabetes", load_diabetes()),
    ("Digits", load_digits()),
    ("Wine", load_wine())
]

# Train and evaluate multiple machine learning models on each dataset
models = [
    ("Decision Tree", DecisionTreeClassifier()),
    ("Logistic Regression", LogisticRegression()),
    ("K-Nearest Neighbors", KNeighborsClassifier()),
    ("Support Vector Machine", SVC()),
    ("Random Forest", RandomForestClassifier())
]

for dataset_name, dataset in datasets:
    print(f"Dataset: {dataset_name}\n")
    
    X, y = dataset.data, dataset.target
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    for model_name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{model_name} Accuracy: {accuracy:.2f}")
        
    print("\n")

# Dataset: Iris
# Decision Tree Accuracy: 1.00
# Logistic Regression Accuracy: 1.00
# K-Nearest Neighbors Accuracy: 1.00
# Support Vector Machine Accuracy: 1.00
# Random Forest Accuracy: 1.00

# Dataset: Diabetes
# Decision Tree Accuracy: 0.01
# Logistic Regression Accuracy: 0.01
# K-Nearest Neighbors Accuracy: 0.00
# Support Vector Machine Accuracy: 0.01
# Random Forest Accuracy: 0.00

# Dataset: Breast Cancer
# Decision Tree Accuracy: 0.92
# Logistic Regression Accuracy: 0.97
# K-Nearest Neighbors Accuracy: 0.96
# Support Vector Machine Accuracy: 0.94
# Random Forest Accuracy: 0.96

# Dataset: Digits
# Decision Tree Accuracy: 0.85
# Logistic Regression Accuracy: 0.96
# K-Nearest Neighbors Accuracy: 0.99
# Support Vector Machine Accuracy: 0.99
# Random Forest Accuracy: 0.98

# Dataset: Wine
# Decision Tree Accuracy: 0.96
# Logistic Regression Accuracy: 0.98
# K-Nearest Neighbors Accuracy: 0.74
# Support Vector Machine Accuracy: 0.76
# Random Forest Accuracy: 1.00