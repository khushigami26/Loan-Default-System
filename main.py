from flask import Blueprint, render_template, redirect, url_for, request, flash, jsonify, current_app
from flask_login import login_required, current_user
from models import PredictionHistory
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import warnings

main = Blueprint('main', __name__)

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
    # Dataset overview (Provided by User)
    original_records = 255347
    original_columns = 18
    after_cleaning = 225694
    removed_records = 29653
    features_used = 17
    
    # Model Accuracies (Provided by User)
    lr_manual_acc = 0.8840
    lr_sklearn_acc = 0.88515 # Best model
    dt_acc = 0.80279
    rf_acc = 0.88488
   
    not_default_pct = round(225694 / 255347 * 100, 1)
    default_pct = round(29653 / 255347 * 100, 1)
    
    return render_template(
        "dashboard.html",
        original_records=f"{original_records:,}",
        original_features=original_columns,
        after_cleaning=f"{after_cleaning:,}",
        features_count=features_used,
        removed_records=f"{removed_records:,}",
        best_model_name="Logistic Regression",
        best_model_accuracy=lr_sklearn_acc,
        lr_manual_accuracy=lr_manual_acc,
        lr_sklearn_accuracy=lr_sklearn_acc,
        rf_accuracy=rf_acc,
        dt_accuracy=dt_acc,
        not_default_percent=not_default_pct,
        default_percent=default_pct,
    )

@main.route("/loan-default/model-performance")
@login_required
def loan_default_model_performance():
    lr_sklearn_acc = 0.88515
    rf_acc = 0.88488
    lr_manual_acc = 0.8840
    dt_acc = 0.80279
    
    return render_template(
        "model_performance.html",
        lr_sklearn_accuracy=lr_sklearn_acc,
        rf_accuracy=rf_acc,
        lr_manual_accuracy=lr_manual_acc,
        dt_accuracy=dt_acc
    )

@main.route("/loan-default/feature-insights")
@login_required
def loan_default_feature_insights():
    return render_template("feature_insights.html")

@main.route("/api/dashboard-data")
@login_required
def dashboard_data():
    total_live_apps = PredictionHistory.objects.count()
    rejected_count = PredictionHistory.objects(prediction__icontains='Default').count()
    approved_count = total_live_apps - rejected_count
    
    avg_income_val = PredictionHistory.objects.average('income') or 0
    avg_loan_val = PredictionHistory.objects.average('loan_amount') or 0

    summary = {
        "rejected_loans": rejected_count,
        "total_applications": total_live_apps,
        "avg_income": round(avg_income_val, 2),
        "approved_loans": approved_count,
        "avg_credit_amount": round(avg_loan_val, 2),
    }

    loan_status_by_gender = {
        "labels": ["House/apartme...", "With parents", "Municipal apartme...", "Rented apartme...", "Office apartme...", "Co-op apartme.."],
        "series": [
            {"label": "Approved", "data": [91.62, 88.20, 91.01, 87.45, 93.35, 91.65]},
            {"label": "Rejected", "data": [8.38, 11.80, 8.99, 12.55, 6.65, 8.35]},
        ],
    }

    applicants_by_education = {
        "labels": ["Secondary / se...", "Higher education", "Other", "Other2"],
        "data": [69, 26, 3, 2],
    }

    applicants_by_income_type = {
        "labels": ["Working", "Commercial as...", "State serva..."],
        "data": [63, 28, 9],
    }

    credit_amount_distribution = {
        "labels": ["0M", "1M", "2M", "3M", "4M"],
        "data": [22000, 18000, 15000, 12000, 8000],
    }

    top_occupations = {
        "labels": [
            "Business Entity...", "Self-employed", "Other", "Medicine",
            "Business Entity Type 2", "Government", "School", "Trade: type 7",
            "Kindergarten", "Construction", "Human Resources", "Cooking",
            "Education", "Banking", "Insurance"
        ],
        "primary": [17.1, 15.8, 12.5, 10.2, 9.5, 8.3, 7.1, 6.2, 5.4, 4.8, 4.0, 3.5, 3.0, 2.5, 2.0],
        "secondary": [1.8, 1.5, 1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.2, 0.1, 0.1],
    }

    return jsonify(
        {
            "summary": summary,
            "loan_status_by_gender": loan_status_by_gender,
            "applicants_by_education": applicants_by_education,
            "applicants_by_income_type": applicants_by_income_type,
            "credit_amount_distribution": credit_amount_distribution,
            "top_occupations": top_occupations,
        }
    )

@main.route("/loan", methods=["GET", "POST"])
@login_required
def loan():
    model = getattr(current_app, 'ml_model', None)
    if model is None:
        return "Model not loaded. Check model file."

    if request.method == "POST":
        try:
            feature_names = getattr(model, "feature_names_in_", None)
            if feature_names is None:
                flash("Model does not expose expected feature names.", "error")
                return render_template("loan_form.html")

            features_dict = {name: 0 for name in feature_names}
            numeric_fields = ["Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed", "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio"]
            
            for f in numeric_fields:
                val = request.form.get(f)
                if val is None or val == "":
                    flash(f"Missing numeric field: {f}", "error")
                    return render_template("loan_form.html")
                try:
                    features_dict[f] = float(val)
                except ValueError:
                    flash(f"Invalid numeric value for: {f}", "error")
                    return render_template("loan_form.html")

            features_dict["HasMortgage"] = 1 if request.form.get("HasMortgage") else 0
            features_dict["HasDependents"] = 1 if request.form.get("HasDependents") else 0
            features_dict["HasCoSigner"] = 1 if request.form.get("HasCoSigner") else 0

            for cat, field in [("Education", "Education"), ("EmploymentType", "EmploymentType"), ("MaritalStatus", "MaritalStatus"), ("LoanPurpose", "LoanPurpose")]:
                val = request.form.get(field)
                if val:
                    key = f"{cat}_{val}"
                    if key in features_dict:
                        features_dict[key] = 1
                else:
                    flash(f"Please select {field}", "error")
                    return render_template("loan_form.html")

            model_type = request.form.get("model_type", "lr_sklearn")
            model_mapping = {
                "lr_sklearn": "Logistic Regression (Sklearn)",
                "rf": "Random Forest Classifier",
                "lr_manual": "Custom Logistic Regression",
                "dt": "Decision Tree Classifier"
            }
            model_name = model_mapping.get(model_type, "Selected Model")

            ordered = [features_dict[name] for name in feature_names]
            # Use DataFrame with named columns to avoid sklearn feature name warning
            features_df = pd.DataFrame([ordered], columns=list(feature_names))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prediction = model.predict(features_df)[0]
            result = "Loan Will Default \u274C" if prediction == 1 else "Loan Approved \u2705"

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

            return render_template("result.html", result=result, model_name=model_name, features=features_dict)

        except Exception as e:
            flash(f"Prediction Error: {e}", "error")
            return render_template("loan_form.html")

    return render_template("loan_form.html")

@main.route("/profile")
@login_required
def profile():
    return render_template("profile.html")

@main.route("/prediction-history")
@login_required
def prediction_history():
    two_days_ago = datetime.utcnow() - timedelta(days=2)
    PredictionHistory.objects(user_id=current_user.pk, date__lt=two_days_ago).delete()
    predictions = PredictionHistory.objects(user_id=current_user.pk).order_by('-date')
    return render_template("prediction_history.html", predictions=predictions)
