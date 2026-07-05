import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def load_data(filepath):
    return pd.read_csv(filepath)

def remove_duplicates(df):
    return df.drop_duplicates()

def handle_missing_values(df):
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    return df

def handle_outliers(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'target' in num_cols:
        num_cols.remove('target')
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)
    return df

def feature_scaling(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'target' in num_cols:
        num_cols.remove('target')
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

def preprocess(input_path, output_path):
    df = load_data(input_path)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = handle_outliers(df)
    df = feature_scaling(df)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Preprocessing selesai. File disimpan di: {output_path}")
    return df

if __name__ == "__main__":
    preprocess(
        input_path="heart_disease_raw/heart.csv",
        output_path="preprocessing/heart_disease_preprocessing/heart_preprocessed.csv"
    )
