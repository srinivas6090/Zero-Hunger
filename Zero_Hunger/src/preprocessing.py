# src/preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean_data(df):
    """Handle missing values and clean the data."""
    df = df.fillna(df.mean())  # Simple imputation
    return df

def feature_engineering(df):
    """Generate new features or transform existing ones."""
    # Example: One-hot encode categorical columns
    df = pd.get_dummies(df, columns=['crop_type'], drop_first=True)
    return df

def split_and_scale_data(df, target_column):
    """Split data and scale features."""
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test
