import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------------------------
# Student Performance Predictor
# Goal: Predict if a student will pass based on exam scores and background info
# -----------------------------------------------

# Load dataset
df = pd.read_csv(r"C:\Users\saraye tel\OneDrive\Desktop\ARCH_Roadmap\Phase_1\StudentsPerformance.csv")

# -----------------------------------------------
# Data Exploration (optional)
# -----------------------------------------------
# print(df.head())
# print(df.info())
# print(df.isnull().sum())

# -----------------------------------------------
# Define Pass/Fail Label BEFORE Normalizing
# -----------------------------------------------
df["Result"] = df["math score"].apply(lambda x: "Pass" if x >= 50 else "Fail")

# -----------------------------------------------
# Categorical Encoding
# -----------------------------------------------
df = pd.get_dummies(df, columns=[
    "gender", "race/ethnicity", "parental level of education",
    "lunch", "test preparation course"
], drop_first=True)

# -----------------------------------------------
# Normalize Scores AFTER Creating the Label
# -----------------------------------------------
numerical = ["math score", "reading score", "writing score"]
df[numerical] = (df[numerical] - df[numerical].mean()) / df[numerical].std()

# -----------------------------------------------
# Split Features & Labels
# -----------------------------------------------
X = df.drop("Result", axis=1).values
y = df["Result"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------
# Model Training
# -----------------------------------------------
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="linear"),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=200)
}

accuracy_scores = {}

# -----------------------------------------------
# Evaluation Loop
# -----------------------------------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    accuracy_scores[name] = acc

    print(f"\n Model: {name}")
    print(f"Accuracy: {acc:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, predictions))

    # Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fail", "Pass"],
                yticklabels=["Fail", "Pass"])
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# -----------------------------------------------
# Accuracy Comparison Chart
# -----------------------------------------------
plt.figure(figsize=(8, 5))
sns.barplot(x=list(accuracy_scores.keys()), y=list(accuracy_scores.values()), palette="viridis")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
for i, acc in enumerate(accuracy_scores.values()):
    plt.text(i, acc + 0.02, f"{acc:.2f}", ha='center')
plt.tight_layout()
plt.show()
