from flask import Flask
from flask import render_template

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/')
def hello_world():
    return render_template("submit.html")

@app.route('/submit', methods=["POST"])
def submit():
    return render_template("result.html")

if __name__ == "__main__":
    app.run(debug=True, port=8000, threaded=True)