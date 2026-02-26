import os
import pickle
from flask import Flask
from flask_login import LoginManager
from mongoengine import connect
import certifi
from dotenv import load_dotenv
from models import User
from auth import auth as auth_blueprint
from main import main as main_blueprint

load_dotenv()

def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "loan_default_secure_key_2026")

    # MongoDB Atlas connection
    atlas_host = os.environ.get("MONGODB_URL")
    if not atlas_host:
        raise ValueError("MONGODB_URL not found in environment variables!")
        
    connect(host=atlas_host, tlsCAFile=certifi.where())

    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"

    @login_manager.user_loader
    def load_user(user_id):
        return User.objects(pk=user_id).first()

    # Load ML Model — path works both locally and on Render
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "loan_default_model.pkl")
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as file:
            app.ml_model = pickle.load(file)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
    else:
        app.ml_model = None
        print(f"⚠️  Model file not found at {MODEL_PATH}!")

    # Register blueprints
    app.register_blueprint(auth_blueprint)
    app.register_blueprint(main_blueprint)

    return app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Debug mode OFF in production (Render sets RENDER env var)
    is_dev = os.environ.get("RENDER") is None
    app.run(host="0.0.0.0", port=port, debug=is_dev)
