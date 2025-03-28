# src/train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
from preprocessing import clean_data, feature_engineering, split_and_scale_data

def train_model(data_path, target_column='yield'):
    """Train the model and save it."""
    df = pd.read_csv(data_path)
    df = clean_data(df)
    df = feature_engineering(df)
    
    X_train, X_test, y_train, y_test = split_and_scale_data(df, target_column)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Model MAE: {mae}")
    
    # Save model
    joblib.dump(model, "../models/final_model.pkl")
    print("Model saved to ../models/final_model.pkl")

# Run training
if __name__ == "__main__":
    train_model('../data/processed/crop_data.csv', target_column='yield')
