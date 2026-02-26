import pickle
import os

model_path = os.path.join("model", "loan_default_model.pkl")
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    print("Model Type:", type(model))
    if hasattr(model, "coef_"):
        print("Coefficients:", model.coef_)
    if hasattr(model, "feature_names_in_"):
        print("Features:", model.feature_names_in_)
else:
    print("Model not found")
