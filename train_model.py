import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. Read the dataset
df = pd.read_csv("creditcard.csv")

# 2. Split into input (X) and target (y)
X = df.drop("Class", axis=1)
y = df["Class"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Create model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 5. Save model and column names
pickle.dump(model, open("fraud_model.pkl", "wb"))
pickle.dump(list(X.columns), open("model_columns.pkl", "wb"))

print("✅ Model saved as fraud_model.pkl and model_columns.pkl")
