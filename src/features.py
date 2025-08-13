from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_preprocessor(numeric_cols, cat_cols):
    num = StandardScaler(with_mean=False)
    cat = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    return ColumnTransformer([
        ("num", num, numeric_cols),
        ("cat", cat, cat_cols),
    ])
