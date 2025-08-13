import os, pandas as pd, joblib
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from features import build_preprocessor

# point to  UCI file
CSV = "data/uci/student-mat.csv"   
if not os.path.exists(CSV):
    raise FileNotFoundError(f"Put UCI CSV at {CSV}")

sep = ";" if CSV.endswith(".csv") else ","
df = pd.read_csv(CSV, sep=sep)

if not {'G1','G2','G3'}.issubset(df.columns):
    raise ValueError("Expected columns G1,G2,G3")

df['grade_trend'] = df['G2'] - df['G1']

numeric = [c for c in ['G1','G2','absences','studytime','grade_trend'] if c in df.columns]
cat_candidates = ['schoolsup','famsup','paid','activities','higher','internet','romantic',
                  'school','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian']
cat = [c for c in cat_candidates if c in df.columns]

X = df[numeric + cat]
y = df['G3']

models = {
    "linear": LinearRegression(),
    "knn": KNeighborsRegressor(n_neighbors=7),
    "dt": DecisionTreeRegressor(max_depth=6, random_state=42),
    "xgb": XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.06,
        subsample=0.9, colsample_bytree=0.9, random_state=42, tree_method="hist"
    )
}

pre = build_preprocessor(numeric, cat)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = {}
for name, m in models.items():
    pipe = Pipeline([("pre", pre), ("reg", m)])
    cv = cross_validate(pipe, X, y, cv=kf,
                        scoring=("neg_mean_absolute_error","neg_root_mean_squared_error","r2"), n_jobs=-1)
    scores[name] = {
        "MAE": -cv["test_neg_mean_absolute_error"].mean(),
        "RMSE": -cv["test_neg_root_mean_squared_error"].mean(),
        "R2": cv["test_r2"].mean()
    }
print("CV:", scores)

best = min(scores, key=lambda n: scores[n]["RMSE"])
print("Best:", best)

best_pipe = Pipeline([("pre", pre), ("reg", models[best])])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
best_pipe.fit(Xtr, ytr)
yp = best_pipe.predict(Xte)
print(f"Holdout MAE: {mean_absolute_error(yte, yp):.3f}  R2: {r2_score(yte, yp):.3f}")

os.makedirs("models", exist_ok=True)
joblib.dump(best_pipe, "models/uci_regressor.joblib")
print("Saved models/uci_regressor.joblib")
