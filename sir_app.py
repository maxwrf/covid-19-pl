from dotenv import load_dotenv
from flask import Flask, render_template, url_for, send_file, make_response
from source.run import run_simple

import os
import time

app = Flask(__name__)


@app.route('/')
@app.route('/home')
def home():
    t = time.time()
    return render_template('home.html', time=t)


@app.route('/fitted')
def fitted():
    return render_template('fitted.html', title='About')


@app.route('/about')
def about():
    return render_template('about.html', title='About')


@app.route('/basic_plot', methods=['GET'])
def basic_plot():
    try:
        bytes_object = run_simple()

        return send_file(bytes_object,
                         attachment_filename='plot.png',
                         mimetype='image/png'
                         )
    except ValueError:
        make_response(
            'Probably features are out of a reasonable range and a number hits infinity',
            400)


if __name__ == '__main__':
    load_dotenv(dotenv_path='config/.env')
    print(os.getenv("FLASK_APP"))
    print(os.getenv("FLASK_DEBUG"))
    app.run(debug=True)
