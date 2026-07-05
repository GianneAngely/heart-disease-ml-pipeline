import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import sys


def load_data(filepath):
    df = pd.read_csv(filepath)
    return df


def handle_missing_values(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    return df


def remove_duplicates(df):
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df


def handle_outliers(df, numeric_cols):
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df


def encode_categorical(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def feature_scaling(df, target_col, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []
    
    cols_to_scale = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col != target_col and col not in exclude_cols
    ]
    
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    return df, scaler


def preprocess(input_path, output_dir, target_col='target'):
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_data(input_path)
    
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != target_col]
    df = handle_outliers(df, numeric_cols)
    
    df = encode_categorical(df)
    df, _ = feature_scaling(df, target_col)
    
    output_path = os.path.join(output_dir, 'heart_preprocessed.csv')
    df.to_csv(output_path, index=False)
    
    return df


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, 'heart_disease_raw', 'heart.csv')
    output_dir = os.path.join(base_dir, 'preprocessing', 'heart_disease_preprocessing')
    
    result = preprocess(input_path, output_dir)
