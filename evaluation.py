from flask import Flask, request
from flask import render_template

from joblib import load

app = Flask(__name__, template_folder="templates")


@app.route("/")
def hello_world():
    return render_template("submit.html")


@app.route("/submit", methods=["POST"])
def submit():
    comment = request.form.get("comment")
    x_test_tfidf = tfidf.transform([comment])
    y_pred = xgb.predict(x_test_tfidf)
    if y_pred[0] == 1:
        result = "Positive"
    else:
        result = "Negative"
    return render_template(
        "result.html", Result=result, Score=y_pred[0], Comment=comment
    )


if __name__ == "__main__":
    tfidf = load("model/tfidf.pkl")
    xgb = load("model/xgb.pkl")
    app.run(debug=True, port=8000, threaded=True)
