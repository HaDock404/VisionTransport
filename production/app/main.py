from flask import Flask
from flask import render_template

app = Flask(__name__)


@app.route('/', methods=["Get", "POST"])
def home():
    return render_template("index.html")


@app.route('/predict', methods=["Get", "POST"])
def predict_image():
    print_result = "Upload"
    return print_result


if __name__ == '__main__':
    app.run(debug=True, port=5002)
