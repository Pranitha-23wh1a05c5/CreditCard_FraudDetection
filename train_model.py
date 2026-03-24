import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

print("Loading datasets...")

# Load datasets
train = pd.read_csv("data/train_transaction.csv")
identity = pd.read_csv("data/train_identity.csv")

# Merge datasets
df = train.merge(identity, on="TransactionID", how="left")

print("Dataset shape:", df.shape)


# ✅ Select useful features
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


# ✅ Handle missing values (VERY IMPORTANT)
df.fillna(-999, inplace=True)


# ✅ Encode categorical safely
df["ProductCD"] = df["ProductCD"].astype("category").cat.codes


# ✅ Split data (STRATIFIED for imbalance)
X = df[features]
y = df["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y   # 🔥 IMPORTANT
)


# ✅ Handle class imbalance
fraud_count = sum(y_train == 1)
non_fraud_count = sum(y_train == 0)

scale_pos_weight = non_fraud_count / fraud_count

print(f"Fraud cases: {fraud_count}, Non-fraud: {non_fraud_count}")
print(f"Scale_pos_weight: {scale_pos_weight:.2f}")


print("Training XGBoost model...")

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,  # 🔥 KEY FIX
    random_state=42
)

model.fit(X_train, y_train)


# ✅ Predictions
pred = model.predict(X_test)


# ✅ Evaluation
print("\nClassification Report:\n")
print(classification_report(y_test, pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, pred))


# ✅ Save model
joblib.dump(model, "app/ieee_fraud_model.pkl")

print("\nModel saved successfully! 🚀")