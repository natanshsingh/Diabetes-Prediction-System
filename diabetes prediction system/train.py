import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

models = [
    LogisticRegression(max_iter=1000),
    RandomForestClassifier(n_estimators=200),
    XGBClassifier(eval_metric="logloss")
]

best_model = models[1]  # Random Forest (stable + explainable)
best_model.fit(X_train_scaled, y_train)

with open("models.pkl", "wb") as f:
    pickle.dump({
        "model": best_model,
        "scaler": scaler,
        "features": X.columns.tolist()
    }, f)

print("Model saved successfully")