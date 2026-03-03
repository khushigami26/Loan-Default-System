from flask import (
    Blueprint, render_template, redirect, url_for, request, flash, current_app
)
from flask_login import login_required, current_user
from models import PredictionHistory
from datetime import datetime, timedelta
import os
import json
import pandas as pd
import warnings

main = Blueprint('main', __name__)


def _load_metadata():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    metadata_path = os.path.join(base_dir, "model", "metadata.json")
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # Fallback to current hard-coded values if metadata is missing
        # or invalid
        return {
            "dataset": {
                "original_records": 255347,
                "original_columns": 18,
                "after_cleaning": 225694,
                "removed_records": 29653,
                "features_used": 17,
            },
            "models": {
                "lr_manual_acc": 0.8840,
                "lr_sklearn_acc": 0.88515,
                "dt_acc": 0.80279,
                "rf_acc": 0.88488,
                "best_model_name": "Logistic Regression",
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

    lr_manual_acc = models.get("lr_manual_acc", 0.0)
    lr_sklearn_acc = models.get("lr_sklearn_acc", 0.0)
    dt_acc = models.get("dt_acc", 0.0)
    rf_acc = models.get("rf_acc", 0.0)
    best_model_name = models.get("best_model_name", "Logistic Regression")

    not_default_pct = round(after_cleaning / original_records * 100, 1) \
        if original_records else 0
    default_pct = round(removed_records / original_records * 100, 1) \
        if original_records else 0
    return render_template(
        "dashboard.html",
        original_records=f"{original_records:,}",
        original_features=original_columns,
        after_cleaning=f"{after_cleaning:,}",
        features_count=features_used,
        removed_records=f"{removed_records:,}",
        best_model_name=best_model_name,
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
    metadata = _load_metadata()
    models = metadata.get("models", {})

    lr_sklearn_acc = models.get("lr_sklearn_acc", 0.0)
    rf_acc = models.get("rf_acc", 0.0)
    lr_manual_acc = models.get("lr_manual_acc", 0.0)
    dt_acc = models.get("dt_acc", 0.0)

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
            numeric_fields = [
                "Age", "Income", "LoanAmount", "CreditScore",
                "MonthsEmployed", "NumCreditLines", "InterestRate",
                "LoanTerm", "DTIRatio"
            ]

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

            features_dict["HasMortgage"] = (
                1 if request.form.get("HasMortgage") else 0
            )
            features_dict["HasDependents"] = (
                1 if request.form.get("HasDependents") else 0
            )
            features_dict["HasCoSigner"] = (
                1 if request.form.get("HasCoSigner") else 0
            )

            for cat, field in [
                ("Education", "Education"),
                ("EmploymentType", "EmploymentType"),
                ("MaritalStatus", "MaritalStatus"),
                ("LoanPurpose", "LoanPurpose")
            ]:
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
            features_df = pd.DataFrame([ordered], columns=list(feature_names))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prediction = model.predict(features_df)[0]
            result = ("Loan Will Default \u274C" if prediction == 1
                      else "Loan Approved \u2705")

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
                model_name=model_name, features=features_dict
            )

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
    days = int(os.environ.get("PREDICTION_HISTORY_DAYS", "30"))
    cutoff = datetime.utcnow() - timedelta(days=days)
    PredictionHistory.objects(
        user_id=current_user.pk, date__lt=cutoff
    ).delete()
    predictions = PredictionHistory.objects(
        user_id=current_user.pk
    ).order_by('-date')
    return render_template(
        "prediction_history.html", predictions=predictions
    )
