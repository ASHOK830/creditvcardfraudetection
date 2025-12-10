import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import pickle

# 1. Read the dataset
df = pd.read_csv("creditcard.csv")

# 2. Split into input (X) and target (y)
X = df.drop("Class", axis=1)
y = df["Class"]

# 3. Train-test split (stratify keeps same fraud ratio in train & test)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


model = LogisticRegression(
    max_iter=500,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("📊 Evaluation on Test Set (Logistic Regression)")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\n🧮 Confusion Matrix [ [TN FP] [FN TP] ]:")
print(cm)

print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred, digits=4))


pickle.dump(model, open("fraud_model.pkl", "wb"))
pickle.dump(list(X.columns), open("model_columns.pkl", "wb"))

print("\n✅ Logistic Regression model saved as fraud_model.pkl and model_columns.pkl")
