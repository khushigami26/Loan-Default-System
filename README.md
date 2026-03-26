# 🏦 Loan Default Prediction System

> A full-stack Machine Learning web application that predicts loan default risk using ensemble ML algorithms, built with Flask and deployed on Render.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [ML Models & Accuracy](#-ml-models--accuracy)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Environment Variables](#-environment-variables)
- [Deployment](#-deployment-on-render)
- [Screenshots](#-screenshots)

---

## 🎯 Overview

The **Loan Default Prediction System** is an AI-powered web platform that helps financial institutions assess the risk of loan applicants defaulting on their loans. It uses multiple classification algorithms trained on 255,000+ real-world records to deliver accurate predictions with an interactive, professional dashboard.

---

## 🌐 Live Demo

🔗 **[loan-default-system.onrender.com](https://loan-default-system.onrender.com)**

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔐 **Secure Authentication** | Register/Login with bcrypt password hashing |
| 🤖 **AI Loan Assessment** | Predict default risk from 16 financial features |
| 🎛️ **4 ML Algorithms** | Choose between Logistic Regression, Random Forest, Decision Tree, Custom LR |
| 📊 **Analytics Dashboard** | Model performance, confusion matrix, accuracy comparison charts |
| 🧠 **AI Risk Insights** | Personalized risk factor analysis per prediction |
| 📜 **Prediction History** | View, track and analyze all past assessments |
| 🌙 **Dark / Light Mode** | Theme toggle persisted across sessions |
| 📱 **Fully Responsive** | Works on desktop, tablet, and mobile |
| 🔬 **Feature Insights** | Correlation heatmap & feature importance analysis |
| ☁️ **Cloud Deployed** | Hosted on Render with MongoDB Atlas |

---

## 🛠️ Tech Stack

### Backend
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green?logo=mongodb)
![Gunicorn](https://img.shields.io/badge/Gunicorn-WSGI-orange)

### Machine Learning
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![NumPy](https://img.shields.io/badge/NumPy-1.x-blue?logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-2.x-blue?logo=pandas)

### Frontend
![HTML5](https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-ES6-yellow?logo=javascript)
![Chart.js](https://img.shields.io/badge/Chart.js-FF6384?logo=chartdotjs&logoColor=white)

---

## 🤖 ML Models & Accuracy

| Model | Accuracy | Status |
|---|---|---|
| ✅ Custom Logistic Regression | **88.47%** | **Best Model** |
| Decision Tree Classifier | 77.06% | — |
| Random Forest Classifier | 76.59% | — |
| Sklearn Logistic Regression | 68.74% | — |

**Best Model Metrics (Custom Logistic Regression):**

| Metric | Value |
|---|---|
| Accuracy | 88.47% |
| Precision | 50.1% |
| Recall | 5.2% |
| F1 Score | 9.4% |

---

## 📦 Dataset

| Property | Value |
|---|---|
| Original Records | 255,347 |
| After Cleaning | 225,694 (29,653 duplicates/NaN removed) |
| Features Used | 24 (16 raw + 8 encoded) |
| Target Variable | `LoanStatus` (0 = No Default, 1 = Default) |
| Class Balance | 88.4% No Default / 11.6% Default |
| Train / Test Split | 80% / 20% (Stratified) |

**Preprocessing Steps:**
- Removed duplicate rows and dropped `LoanID`
- Applied **One-Hot Encoding** on: `Education`, `EmploymentType`, `MaritalStatus`, `LoanPurpose`
- Stratified 80/20 train-test split

**Input Features:**

```
Age, Income, LoanAmount, CreditScore, MonthsEmployed,
NumCreditLines, InterestRate, LoanTerm, DTIRatio,
Education (High School, Master's, PhD), 
EmploymentType (Part-time, Self-employed, Unemployed),
MaritalStatus (Married, Single), 
LoanPurpose (Business, Education, Home, Other),
HasMortgage, HasDependents, HasCoSigner
```

---

## 📁 Project Structure

```
loan_default/
├── app.py                  # Flask app factory & entry point
├── auth.py                 # Login, Register, Logout routes
├── main.py                 # Dashboard, Prediction, Profile routes
├── models.py               # MongoEngine User & PredictionHistory models
├── requirements.txt        # Python dependencies
├── Procfile                # Gunicorn start command for Render
├── render.yaml             # Render deployment config
├── model/
│   └── loan_default_model.pkl   # Trained ML model (pickle)
├── static/
│   ├── style.css
│   ├── dashboard.css
│   ├── loan_form.css
│   ├── result.css
│   ├── profile.css
│   ├── prediction_history.css
│   ├── model_performance.css
│   ├── home_page.css
│   ├── login.css
│   └── register.css
└── templates/
    ├── base.html
    ├── home_page.html
    ├── login.html
    ├── register.html
    ├── sidebar.html
    ├── loan_form.html
    ├── result.html
    ├── dashboard.html
    ├── model_performance.html
    ├── feature_insights.html
    ├── prediction_history.html
    ├── profile.html
    └── toast.html
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip
- MongoDB Atlas account (free tier)

### 1. Clone the Repository

```bash
git clone https://github.com/khushigami262/loan-default-system.git
cd loan-default-system
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables

Create a `.env` file or export:

```bash
export SECRET_KEY="your_secret_key_here"
export MONGODB_URL="mongodb+srv://<user>:<pass>@cluster.mongodb.net/ML"
```

### 5. Run the App

```bash
python app.py
```

Visit: **http://127.0.0.1:5000**

---

## 🔐 Environment Variables

| Variable | Description | Required |
|---|---|---|
| `SECRET_KEY` | Flask session secret key | ✅ Yes |
| `MONGODB_URL` | MongoDB Atlas connection string | ✅ Yes |

---

## ☁️ Deployment on Render

1. Push the repo to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect GitHub → Select this repo
4. Render auto-detects `render.yaml`
5. Add environment variables in Render dashboard:
   - `SECRET_KEY` → any long random string
   - `MONGODB_URL` → your Atlas connection string
6. Click **Deploy** 🚀

**Start command used:**
```bash
gunicorn app:app --workers 2 --threads 2 --timeout 120
```

---

## 📸 Screenshots

| Page | Description |
|---|---|
| 🏠 Home | Landing page with feature highlights |
| 🔐 Login/Register | Secure authentication with password toggle |
| 📝 Loan Form | 16-feature input form with model selection |
| ⏳ Loading | Animated multi-ring AI spinner |
| ✅ Result | Prediction result with AI risk insights |
| 📊 Dashboard | Charts, model comparison, dataset stats |
| 🔬 Model Performance | Confusion matrix + accuracy metrics |
| 🧠 Feature Insights | Correlation heatmap, feature importance |
| 📜 History | Expandable prediction history table |
| 👤 Profile | User info display |

---

## 👩‍💻 Author

**Khushi** — [@khushigami262](https://github.com/khushigami262)

---

## 📄 License

This project is for academic purposes.

---

<p align="center">🚀 Built with using Flask & scikit-learn</p>
