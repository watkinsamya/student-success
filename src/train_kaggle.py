
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from features import build_preprocessor



# Load Kaggle CSV
CSV = "data/kaggle/StudentsPerformance.csv"  
if not os.path.exists(CSV):
    raise FileNotFoundError(f"Put Kaggle CSV at {CSV}")

df = pd.read_csv(CSV)

# Normalize common column names
df = df.rename(
    columns={
        "race/ethnicity": "race_ethnicity",
        "parental level of education": "parent_education",
        "test preparation course": "test_prep",
        "math score": "math",
        "reading score": "reading",
        "writing score": "writing",
    }
)




df["avg_score"] = df[["math", "reading", "writing"]].mean(axis=1)


def bucket(x: float) -> str:
    if x < 60:
        return "At-Risk"
    if x < 80:
        return "Average"
    return "High"


df["risk"] = df["avg_score"].apply(bucket)


numeric = ["math", "reading", "writing"]
cat = [c for c in ["gender", "race_ethnicity", "parent_education", "lunch", "test_prep"] if c in df.columns]

X = df[numeric + cat].copy()
y = df["risk"].copy()


le = LabelEncoder()
y_enc = le.fit_transform(y)  # 'At-Risk','Average','High' -> 0/1/2

# Preprocessor (scaler + one-hot) defined in src/features.py
pre = build_preprocessor(numeric, cat)


models = {
    "logreg": LogisticRegression(max_iter=1000),  
    "knn": KNeighborsClassifier(n_neighbors=7),
    "dt": DecisionTreeClassifier(max_depth=6, random_state=42),
    "xgb": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        tree_method="hist",
        objective="multi:softprob",
        num_class=len(le.classes_),
    ),
}


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = {}
for name, m in models.items():
    pipe = Pipeline([("pre", pre), ("clf", m)])
    cv = cross_validate(pipe, X, y_enc, cv=skf, scoring=["accuracy", "f1_macro"], n_jobs=-1)
    scores[name] = {k.replace("test_", ""): v.mean() for k, v in cv.items() if k.startswith("test_")}

print("CV:", scores)
best = max(scores, key=lambda n: scores[n]["f1_macro"])
print("Best:", best)


best_pipe = Pipeline([("pre", pre), ("clf", models[best])])
Xtr, Xte, ytr, yte = train_test_split(X, y_enc, test_size=0.2, stratify=y_enc, random_state=42)
best_pipe.fit(Xtr, ytr)
pred_enc = best_pipe.predict(Xte)

print("\nHoldout report:\n", classification_report(yte, pred_enc, target_names=le.classes_))


Path("models").mkdir(exist_ok=True)
joblib.dump({"pipeline": best_pipe, "label_encoder": le}, "models/kaggle_classifier.joblib")
print("Saved models/kaggle_classifier.joblib (dict with 'pipeline' and 'label_encoder')")


Path("reports/figures").mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(5.5, 5))
ConfusionMatrixDisplay.from_predictions(
    yte, pred_enc, display_labels=le.classes_, xticks_rotation=45, ax=ax
)
ax.set_title("Kaggle Risk – Confusion Matrix (Holdout)")
fig.tight_layout()
out_path = Path("reports/figures/kaggle_confusion_matrix.png")
fig.savefig(out_path, dpi=150)
print(f"Saved {out_path}")
