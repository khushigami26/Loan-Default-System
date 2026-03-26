from flask import (
    Blueprint, render_template, redirect, url_for, request, flash, current_app
)
from flask_login import login_required, current_user
from models import PredictionHistory
from datetime import datetime, timedelta
import os
import json
import pandas as pd
import numpy as np
import joblib
import warnings

main = Blueprint('main', __name__)


def _load_metadata():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    metadata_path = os.path.join(base_dir, "model", "metadata.json")
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "dataset": {
                "original_records": 255347,
                "original_columns": 18,
                "after_cleaning": 225694,
                "removed_records": 29653,
                "features_used": 24,
            },
            "best_model_name": "Random Forest Classifier",
            "models": {
                "lr_manual": {"accuracy": 0.8784, "precision": 0.4845, "recall": 0.8237, "f1_score": 0.6102},
                "lr_sklearn": {"accuracy": 0.6874, "precision": 0.2231, "recall": 0.6873, "f1_score": 0.3367},
                "dt": {"accuracy": 0.7706, "precision": 0.2335, "recall": 0.4319, "f1_score": 0.3031},
                "rf": {"accuracy": 0.7659, "precision": 0.2456, "recall": 0.4956, "f1_score": 0.3284},
            },
        }


@main.route("/")
def home():
    return render_template("home_page.html")


@main.route("/dashboard")
@login_required
def dashboard():
    return redirect(url_for("main.loan_default_dashboard"))


@main.route("/loan-default")
@login_required
def loan_default_dashboard():
    metadata = _load_metadata()
    dataset = metadata.get("dataset", {})
    models = metadata.get("models", {})

    original_records = dataset.get("original_records", 0)
    original_columns = dataset.get("original_columns", 0)
    after_cleaning = dataset.get("after_cleaning", 0)
    removed_records = dataset.get("removed_records", 0)
    features_used = dataset.get("features_used", 0)
    best_model_name = metadata.get("best_model_name", "Custom Logistic Regression")

    target = dataset.get("target_distribution", {})
    not_default_pct = target.get("not_default_percent", 0.0)
    default_pct = target.get("default_percent", 0.0)

    # Classification Metrics Mapping
    model_metrics = {}
    for m_id, m_data in models.items():
        model_metrics[m_id] = {
            "accuracy": m_data.get("accuracy", 0.0),
            "precision": m_data.get("precision", 0.0),
            "recall": m_data.get("recall", 0.0),
            "f1": m_data.get("f1_score", 0.0)
        }

    best_model_accuracy = model_metrics.get("rf", {}).get("accuracy", 0.0)
    if best_model_name == "Custom Logistic Regression":
        best_model_accuracy = model_metrics.get("lr_manual", {}).get("accuracy", 0.0)

    return render_template(
        "dashboard.html",
        original_records=f"{original_records:,}",
        original_features=original_columns,
        after_cleaning=f"{after_cleaning:,}",
        features_count=features_used,
        removed_records=f"{removed_records:,}",
        best_model_name=best_model_name,
        best_model_accuracy=best_model_accuracy,
        metrics=model_metrics,
        not_default_percent=not_default_pct,
        default_percent=default_pct,
    )


@main.route("/loan-default/model-performance")
@login_required
def loan_default_model_performance():
    metadata = _load_metadata()
    models = metadata.get("models", {})
    metrics = {}
    for m_id, m_data in models.items():
        metrics[m_id] = {
            "accuracy": m_data.get("accuracy", 0.0),
            "precision": m_data.get("precision", 0.0),
            "recall": m_data.get("recall", 0.0),
            "f1": m_data.get("f1_score", 0.0)
        }

    return render_template(
        "model_performance.html",
        metrics=metrics
    )


@main.route("/loan-default/feature-insights")
@login_required
def loan_default_feature_insights():
    metadata = _load_metadata()
    models = metadata.get("models", {})
    lr_manual_acc = models.get("lr_manual", {}).get("accuracy", 0.0)

    return render_template(
        "feature_insights.html",
        lr_manual_accuracy=lr_manual_acc
    )


@main.route("/loan", methods=["GET", "POST"])
@login_required
def loan():
    # Check if any specialized models are loaded
    models = getattr(current_app, 'ml_models', {})
    if not models:
        return "Model not loaded. Check model file."

    metadata = _load_metadata()
    models_metadata = metadata.get("models", {})
    
    feature_names = [
        'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 
        'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio',
        'Education_High School', 'Education_Master\'s', 'Education_PhD',
        'EmploymentType_Part-time', 'EmploymentType_Self-employed', 'EmploymentType_Unemployed',
        'MaritalStatus_Married', 'MaritalStatus_Single',
        'HasMortgage_Yes', 'HasDependents_Yes',
        'LoanPurpose_Business', 'LoanPurpose_Education', 'LoanPurpose_Home', 'LoanPurpose_Other',
        'HasCoSigner_Yes'
    ]
    
    # Classification Metrics Mapping
    metrics = {}
    for m_id, m_data in models_metadata.items():
        metrics[m_id] = {
            "accuracy": m_data.get("accuracy", 0.0),
            "precision": m_data.get("precision", 0.0),
            "recall": m_data.get("recall", 0.0),
            "f1": m_data.get("f1_score", 0.0)
        }

    context = {
        "metrics": metrics,
        "best_id": "lr_manual" if metadata.get("best_model_name", "Custom Logistic Regression") == "Custom Logistic Regression" else "rf"
    }

    if request.method == "POST":
        try:
            features_dict = {name: 0.0 for name in feature_names}
            
            numeric_fields = [
                "Age", "Income", "LoanAmount", "CreditScore",
                "MonthsEmployed", "NumCreditLines", "InterestRate",
                "LoanTerm", "DTIRatio"
            ]
            for f in numeric_fields:
                val = request.form.get(f)
                if val:
                    try:
                        features_dict[f] = float(val)
                    except ValueError:
                        flash(f"Invalid numeric value for: {f}", "error")
                        return render_template("loan_form.html", **context)

            # Encode categorical features manually based on the model's dummy columns
            # Values that were 'dropped' during training (first category alphabetically) are implicit
            
            # Education_High School, Education_Master's, Education_PhD
            edu = request.form.get("Education")
            features_dict["Education_High School"] = 1.0 if edu == "High School" else 0.0
            features_dict["Education_Master's"] = 1.0 if edu == "Master's" else 0.0
            features_dict["Education_PhD"] = 1.0 if edu == "PhD" else 0.0

            # EmploymentType_Part-time, EmploymentType_Self-employed, EmploymentType_Unemployed
            emp = request.form.get("EmploymentType")
            features_dict["EmploymentType_Part-time"] = 1.0 if emp == "Part-time" else 0.0
            features_dict["EmploymentType_Self-employed"] = 1.0 if emp == "Self-employed" else 0.0
            features_dict["EmploymentType_Unemployed"] = 1.0 if emp == "Unemployed" else 0.0

            # MaritalStatus_Married, MaritalStatus_Single
            marital = request.form.get("MaritalStatus")
            features_dict["MaritalStatus_Married"] = 1.0 if marital == "Married" else 0.0
            features_dict["MaritalStatus_Single"] = 1.0 if marital == "Single" else 0.0

            # Binary Fields (Correctly handles 'on' from web checkboxes)
            features_dict["HasMortgage_Yes"] = 1.0 if request.form.get("HasMortgage") else 0.0
            features_dict["HasDependents_Yes"] = 1.0 if request.form.get("HasDependents") else 0.0
            features_dict["HasCoSigner_Yes"] = 1.0 if request.form.get("HasCoSigner") else 0.0

            # LoanPurpose_Business, LoanPurpose_Education, LoanPurpose_Home, LoanPurpose_Other
            purpose = request.form.get("LoanPurpose")
            features_dict["LoanPurpose_Business"] = 1.0 if purpose == "Business" else 0.0
            features_dict["LoanPurpose_Education"] = 1.0 if purpose == "Education" else 0.0
            features_dict["LoanPurpose_Home"] = 1.0 if purpose == "Home" else 0.0
            features_dict["LoanPurpose_Other"] = 1.0 if purpose == "Other" else 0.0

            ordered_values = [features_dict[name] for name in feature_names]

            model_type = request.form.get("model_type", "lr_sklearn")
            model_mapping = {
                "lr_sklearn": "Sklearn Logistic Regression",
                "rf": "Random Forest Classifier",
                "lr_manual": "Custom Logistic Regression",
                "dt": "Decision Tree Classifier"
            }
            model_name = model_mapping.get(model_type, "Selected Model")
            
            # Prepare inference data (Pipeline will handle scaling automatically)
            features_df = pd.DataFrame([ordered_values], columns=feature_names)
            
            # Select the correct loaded engine
            chosen_engine = current_app.ml_models.get(model_type)
            if not chosen_engine:
                chosen_engine = getattr(current_app, "ml_model", None)
            
            if not chosen_engine:
                flash("Selected prediction engine not initialized on server.", "error")
                return render_template("loan_form.html", **context)
                
            # Perform Prediction
            if isinstance(chosen_engine, dict) and chosen_engine.get('type') == 'manual_logistic_numpy':
                # Manual Dot Product Math (NumPy implementation)
                weights = chosen_engine.get('weights')
                bias = chosen_engine.get('bias')
                
                # Apply scaling (Manual model expects scaled data)
                features_scaled = current_app.scaler.transform(features_df)
                
                # Equation: z = sum(w*x) + b
                z = np.dot(features_scaled, weights) + bias
                
                # Activation: sigmoid(z) = 1 / (1 + e^-z)
                prediction_proba = float(1 / (1 + np.exp(-z[0])))
                prediction = 1 if prediction_proba >= 0.5 else 0
            else:
                # Standard Scikit-Learn prediction
                prediction = int(chosen_engine.predict(features_df)[0])
                
                # Get Probability with robust handling
                prediction_proba = 0.5
                if hasattr(chosen_engine, "predict_proba"):
                    try:
                        probs = chosen_engine.predict_proba(features_df)[0]
                        prediction_proba = float(probs[1]) 
                    except:
                        prediction_proba = 0.5
            
            # --- Hybrid Assessment: Model Probability + Financial Sanity Guards ---
            # Calibrated to 0.50 for the high-accuracy custom model
            BANK_THRESHOLD = 0.50
            
            # Start with ML Probability
            is_default = (prediction_proba > BANK_THRESHOLD)
            
            # --- Sanity Guards & Affluence Overrides ---
            loan_val = float(request.form.get("LoanAmount", 0))
            income_val = float(request.form.get("Income", 0))
            credit_val = float(request.form.get("CreditScore", 0))
            
            # 1. Professional Override: High-Net-Worth Individuals (Strong stability)
            # If income is > 1M and they have Fair-to-Good credit, we override ML skepticism
            if income_val >= 1000000 and credit_val >= 550:
                is_default = False
            
            # 2. Strict Loan-to-Income Rule (Still reject if loan is > 20x annual income)
            if income_val > 0 and (loan_val / income_val) > 20:
                is_default = True
            
            # 3. Critical Poverty Rule (Manual check for non-viable income < 35k)
            if income_val < 35000:
                is_default = True
                
            # 4. Hard Credit Floor (Reject under 420 regardless of income)
            if credit_val < 420:
                is_default = True
                
            result = "Risk of Default \u274C" if is_default else "Loan Approved \u2705"
            
            # --- Database Persistence ---
            history = PredictionHistory(
                user_id=current_user.pk,
                loan_amount=features_dict.get("LoanAmount", 0),
                income=features_dict.get("Income", 0),
                model_used=model_name,
                prediction=result,
                credit_score=features_dict.get("CreditScore"),
                interest_rate=features_dict.get("InterestRate"),
                loan_term=features_dict.get("LoanTerm"),
                age=features_dict.get("Age"),
                dti_ratio=features_dict.get("DTIRatio")
            )
            history.save()

            return render_template(
                "result.html", result=result,
                model_name=model_name, used_model_id=model_type,
                features=features_dict, metrics=metrics, 
                best_id=context["best_id"],
                probability=prediction_proba,
                is_default=is_default
            )

        except Exception as e:
            flash(f"Prediction Error: {e}", "error")
            return render_template("loan_form.html", **context)

    return render_template("loan_form.html", **context)


@main.route("/profile")
@login_required
def profile():
    return render_template("profile.html")


@main.route("/prediction-history")
@login_required
def prediction_history():
    predictions = PredictionHistory.objects(
        user_id=current_user.pk
    ).order_by('-date')
    return render_template(
        "prediction_history.html", predictions=predictions
    )
