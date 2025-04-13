from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
MODEL_FILE = "review_model.pkl"

# Train model if not already saved
def train_model():
    df = pd.read_csv("dataset.csv")
    X = df["text"]
    y = df["label"]

    vectorizer = TfidfVectorizer(stop_words='english')
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_vec, y)

    joblib.dump((vectorizer, model), MODEL_FILE)

# Load model or train it
if not os.path.exists(MODEL_FILE):
    train_model()

vectorizer, model = joblib.load(MODEL_FILE)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    review = ""
    if request.method == "POST":
        review = request.form["review"]
        X_vec = vectorizer.transform([review])
        prediction = model.predict(X_vec)[0]
    return render_template("index.html", prediction=prediction, review=review)

if __name__ == "__main__":
    app.run(debug=True)
