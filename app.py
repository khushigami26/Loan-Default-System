import os
import pickle
import certifi
from flask import Flask
from flask_login import LoginManager
from dotenv import load_dotenv
from mongoengine import connect, disconnect
from flask_wtf import CSRFProtect
from models import User
from auth import auth as auth_blueprint
from main import main as main_blueprint

load_dotenv()


def create_app():
    app = Flask(__name__)

    secret_key = os.environ.get("SECRET_KEY")
    app.config["SECRET_KEY"] = secret_key

    # Session security
    app.config.setdefault("SESSION_COOKIE_HTTPONLY", True)  # protect cookies
    app.config.setdefault("SESSION_COOKIE_SAMESITE", "Lax")  # prevent CSRF
    if os.environ.get("RENDER") is not None:
        app.config.setdefault("SESSION_COOKIE_SECURE", True)

    #  protection
    CSRFProtect(app)

    @app.route("/health") 
    def health():
        return {"status": "healthy", "mongodb": "checking..."}, 200

    # MongoDB  connection
    atlas_host = os.environ.get("MONGODB_URL")
    if not atlas_host:
        print("MONGODB_URL environment variable is missing!")

    try:
        disconnect()
        connect(
            host=atlas_host,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=5000,
            uuidRepresentation='standard'
        )
        from models import PredictionHistory
        PredictionHistory.ensure_indexes()
    except Exception as e:
        print(f"MongoDB connection failed: {e}")

    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"

    @login_manager.user_loader
    def load_user(user_id):
        try:
            return User.objects(pk=user_id).first()
        except Exception:
            return None

    # Load Models
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
    
    import joblib
    app.ml_models = {}
    
    # Model Map
    model_files = {
        "rf": "rf_model.pkl",
        "dt": "dt_model.pkl",
        "lr_sklearn": "lr_sklearn_model.pkl",
        "lr_manual": "lr_manual_model.pkl",
    }

    # Load specific model files with individual error handling for better resilience on Render
    for model_id, filename in model_files.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            try:
                app.ml_models[model_id] = joblib.load(path)
                print(f"Model {model_id} successfully loaded from {filename}")
            except Exception as e:
                print(f"CRITICAL: Failed to load {model_id} from {filename}: {e}")
    
    # Check if we have at least one valid model
    if not app.ml_models:
        print("CRITICAL NOTIFICATION: No models were successfully loaded into memory!")
    
    # Fallback for loan_model.pkl (Absolute Legacy/Local Fallback)
    main_path = os.path.join(MODEL_DIR, "loan_model.pkl")
    if os.path.exists(main_path):
        try:
            main_model = joblib.load(main_path)
            app.ml_model = main_model
            if not app.ml_models:
                app.ml_models["rf"] = main_model
                print("Using legacy loan_model.pkl as primary fallback.")
        except Exception:
            app.ml_model = None
    else:
        app.ml_model = None

    print(f"Active Prediction Engines: {list(app.ml_models.keys())}")

    app.register_blueprint(auth_blueprint)
    app.register_blueprint(main_blueprint)

    return app


app = create_app()


@app.route("/health-full")
def health_full():
    db_status = "connected"
    try:
        User.objects.count()
    except Exception as e:
        db_status = f"error: {str(e)}"

    return {
        "status": "online",
        "mongodb": db_status,
        "model_loaded": app.ml_model is not None
    }, 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    is_dev = os.environ.get("RENDER") is None
    app.run(host="0.0.0.0", port=port, debug=is_dev)
