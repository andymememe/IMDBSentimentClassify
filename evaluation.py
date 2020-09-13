from flask import Flask, request
from flask import render_template

app = Flask(__name__, template_folder='templates')

@app.route('/')
def hello_world():
    return render_template("submit.html")

@app.route('/submit', methods=["POST"])
def submit():
    comment = request.form.get('comment')
    return render_template("result.html", Result="Test", Comment=comment)

if __name__ == "__main__":
    app.run(debug=True, port=8000, threaded=True)