# src/predict.py
import joblib
import pandas as pd

def predict(input_data):
    """Load model and make predictions."""
    model = joblib.load('../models/final_model.pkl')
    predictions = model.predict(input_data)
    return predictions

if __name__ == "__main__":
    # Sample data for prediction
    data = pd.DataFrame({
        'temperature': [25],
        'rainfall': [100],
        'soil_quality': [0.8],
        'crop_type_rice': [1],
        'crop_type_wheat': [0]
    })
    
    print("Prediction:", predict(data))
