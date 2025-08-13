import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# This Loads UCI dataset
data_path = Path("data/uci/student-mat.csv")  # updated
df = pd.read_csv(data_path, sep=';')

#  Adding grade trend 
df['grade_trend'] = df['G2'] - df['G1']

#  Features & target 
features = ['G1', 'G2', 'absences', 'studytime', 'grade_trend', 'higher', 'internet', 'romantic']
target = 'G3'

X = df[features]
y = df[target]

#  Categorical & numeric separation 
cat_cols = ['higher', 'internet', 'romantic']
num_cols = ['G1', 'G2', 'absences', 'studytime', 'grade_trend']

# Preprocessor 
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)

# Model 
model = GradientBoostingRegressor(random_state=42)

# Pipeline 
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

# Train/test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipe.fit(X_train, y_train)

# This saves the model
Path("models").mkdir(exist_ok=True)
joblib.dump(pipe, "models/uci_regressor.joblib")

print(" UCI regressor trained and saved to models/uci_regressor.joblib")
