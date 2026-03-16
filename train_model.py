import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Remove missing values
df = df.dropna()

# Convert Churn column to numeric
df["Churn"] = df["Churn"].map({"Yes":1,"No":0})

# Drop customerID column
df = df.drop("customerID", axis=1)

# Convert categorical columns to numeric
df = pd.get_dummies(df, drop_first=True)

# Split features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:,1]

# Evaluation
print(classification_report(y_test, pred))
print("ROC-AUC:", roc_auc_score(y_test, prob))

# Save model
joblib.dump(model, "model/churn_model.pkl")

print("Model saved successfully!")