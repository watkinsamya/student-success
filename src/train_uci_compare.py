import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

# load
df = pd.read_csv("data/uci/student-mat.csv", sep=";")
df["grade_trend"] = df["G2"] - df["G1"]

num = ["G1","G2","absences","studytime","grade_trend"]
cat = ["higher","internet","romantic"]

X = df[num+cat]; y = df["G3"]

pre = ColumnTransformer([
    ("num","passthrough", num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
])

models = {
    "Linear": LinearRegression(),
    "KNN": KNeighborsRegressor(n_neighbors=7),
    "DecisionTree": DecisionTreeRegressor(max_depth=6, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
rows = []
for name, m in models.items():
    pipe = Pipeline([("pre", pre), ("reg", m)])
    cv = cross_validate(pipe, X, y, cv=kf,
                        scoring=("neg_mean_absolute_error","neg_root_mean_squared_error","r2"),
                        n_jobs=-1)
    rows.append({
        "Model": name,
        "MAE": -cv["test_neg_mean_absolute_error"].mean(),
        "RMSE": -cv["test_neg_root_mean_squared_error"].mean(),
        "R2": cv["test_r2"].mean()
    })
res = pd.DataFrame(rows).sort_values("RMSE")
print(res.to_string(index=False))
res.to_csv("reports/uci_cv_metrics.csv", index=False)
