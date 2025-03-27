import yaml
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Load parameters from params.yaml
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

# Load dataset
df = pd.read_csv('students_placement.csv')

# Split features and target
X = df.drop(columns=['placed'])
y = df['placed']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["split"]["test_size"], random_state=params["split"]["random_state"]
)

# Apply scaling if enabled
if params["scaling"]["enabled"]:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

# Select and train model
if params["model"]["type"] == "logistic_regression":
    model = LogisticRegression()
elif params["model"]["type"] == "random_forest":
    model = RandomForestClassifier(
        n_estimators=params["model"]["n_estimators"],
        random_state=params["model"]["random_state"]
    )
elif params["model"]["type"] == "svm":
    model = SVC(kernel=params["model"]["kernel"])
else:
    raise ValueError("Unsupported model type")

# Train model
model.fit(X_train, y_train)

# Evaluate model
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model Accuracy: {accuracy:.4f}")

# Save model
pickle.dump(model, open('model.pkl', 'wb'))
print("Model saved as model.pkl")
