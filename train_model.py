# train_model.py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
df = pd.read_csv("soil_dataset.csv")

# Features and target
X = df[["pH", "soil_type", "temperature", "moisture"]]
y = df["crop"]

# Preprocessing: one-hot soil_type, scale numeric
preproc = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["pH", "temperature", "moisture"]),
        ("soil", OneHotEncoder(handle_unknown="ignore"), ["soil_type"])
    ]
)

model = Pipeline([
    ("preproc", preproc),
    ("clf", RandomForestClassifier(n_estimators=150, random_state=42))
])

# Train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "crop_model.joblib")
print("Saved model to crop_model.joblib")
