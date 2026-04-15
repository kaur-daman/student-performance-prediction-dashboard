import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

data = pd.read_csv("data/student_dataset.csv")

# Handle missing values
num_cols = data.select_dtypes(include=["int64","float64"]).columns
cat_cols = data.select_dtypes(include=["object"]).columns

num_imputer = SimpleImputer(strategy="median")
data[num_cols] = num_imputer.fit_transform(data[num_cols])

cat_imputer = SimpleImputer(strategy="most_frequent")
data[cat_cols] = cat_imputer.fit_transform(data[cat_cols])

# Encoding categorical variables
le = LabelEncoder()

for col in cat_cols:
    data[col] = le.fit_transform(data[col])

data.to_csv("data/clean_dataset.csv",index=False)
