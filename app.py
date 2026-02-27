import os
import pickle
import certifi
from flask import Flask
from flask_login import LoginManager
from dotenv import load_dotenv
from mongoengine import connect, disconnect
from models import User
from auth import auth as auth_blueprint
from main import main as main_blueprint

load_dotenv()


def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY")

    @app.route("/health")
    def health():
        return {"status": "healthy", "mongodb": "checking..."}, 200

    # MongoDB Atlas connection
    atlas_host = os.environ.get("MONGODB_URL")
    if not atlas_host:
        print("MONGODB_URL environment variable is missing!")

    try:
        disconnect()
        connect(
            host=atlas_host,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=5000
        )
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

    # Load ML Model
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "loan_default_model.pkl")
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as file:
                app.ml_model = pickle.load(file)
        else:
            app.ml_model = None
            print(f"Model file not found at {MODEL_PATH}!")
    except Exception as e:
        app.ml_model = None
        print(f"Error loading model: {e}")

    # Register blueprints
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
