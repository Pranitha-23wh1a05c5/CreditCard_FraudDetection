import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print("Loading datasets...")

# Load datasets
train = pd.read_csv("data/train_transaction.csv")
identity = pd.read_csv("data/train_identity.csv")

# Merge datasets
df = train.merge(identity, on="TransactionID", how="left")

print("Dataset shape:", df.shape)

# Select useful features
features = [
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "addr1",
    "addr2",
    "dist1"
]

df = df[features + ["isFraud"]]

# Encode categorical
df["ProductCD"] = df["ProductCD"].astype("category").cat.codes

# Split data
X = df[features]
y = df["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training XGBoost model...")

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print(classification_report(y_test, pred))

# Save model
joblib.dump(model, "app/ieee_fraud_model.pkl")

print("Model saved successfully!")