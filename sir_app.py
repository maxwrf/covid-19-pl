from dotenv import load_dotenv
from flask import Flask, render_template, url_for
from source.run import run_simple

import os

app = Flask(__name__)


@app.route('/')
@app.route('/home')
def home():
    Spreds, Ipreds, Rpreds = run_simple()
    return render_template('home.html', preds=[Spreds, Ipreds, Rpreds])


@app.route('/fitted')
def fitted():
    return render_template('about.html', title='About')


@app.route('/about')
def about():
    return render_template('about.html', title='About')


if __name__ == '__main__':
    load_dotenv(dotenv_path='config/.env')
    print(os.getenv("FLASK_APP"))
    print(os.getenv("FLASK_DEBUG"))
    app.run(debug=True)
