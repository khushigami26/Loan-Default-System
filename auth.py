from flask import Blueprint, render_template, redirect, url_for, request, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, logout_user, login_required
from models import User

auth = Blueprint('auth', __name__)


@auth.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if (len(password) < 8 or
                not any(c.isalpha() for c in password) or
                not any(c.isdigit() for c in password)):
            flash(("Password must be at least 8 characters and include "
                   "letters and numbers."), "error")
            return redirect(url_for("auth.register"))

        existing_user = User.objects(username=username).first()
        if existing_user:
            flash("Username already exists", "error")
            return redirect(url_for("auth.register"))

        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password=hashed_password)
        new_user.save()

        flash("Registration successful. Please login.", "success")
        return redirect(url_for("auth.login", username=username))

    return render_template("register.html")


@auth.route("/login", methods=["GET", "POST"])
def login():
    username_val = request.args.get('username', '')

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = User.objects(username=username).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            flash("Logged in successfully!!", "success")

            return redirect(url_for("main.dashboard"))
        else:
            flash("Invalid username or password", "error")

    return render_template("login.html", username=username_val, password="")


@auth.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out successfully!!", "success")
    return redirect(url_for("main.home"))
