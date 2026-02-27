from mongoengine import Document, StringField, FloatField, ReferenceField, DateTimeField
from flask_login import UserMixin
from datetime import datetime


class User(UserMixin, Document):
    username = StringField(max_length=100, unique=True, required=True)
    password = StringField(max_length=200, required=True)


class PredictionHistory(Document):
    user_id = ReferenceField(User, required=True)
    loan_amount = FloatField(required=True)
    income = FloatField(required=True)
    model_used = StringField(max_length=50, required=True)
    prediction = StringField(max_length=50, required=True)

    # Detailed fields
    credit_score = FloatField()
    interest_rate = FloatField()
    loan_term = FloatField()
    age = FloatField()
    dti_ratio = FloatField()

    date = DateTimeField(default=datetime.utcnow)
