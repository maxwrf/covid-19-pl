from dotenv import load_dotenv
from flask import Flask, render_template, url_for, send_file, make_response, request
from source.run import run_simple, run_fit, load_data
from source.scraper.wiki_pl_scraper import crawl_wiki_pl

import os
import time

app = Flask(__name__)


@app.route('/')
@app.route('/home')
def home():
    if request.args['beta'] == '':
        beta = 0.0005
    else:
        beta = request.args['beta']

    if request.args['gamma'] == '':
        gamma = 0.1
    else:
        gamma = request.args['gamma']

    if request.args['population'] == '':
        population = 1500
    else:
        population = request.args['population']

    if request.args['time'] == '':
        integ_time = 60
    else:
        integ_time = request.args['time']

    t = time.time()  # to prevent browser from caching images

    return render_template('home.html',
                           time=t,
                           b=beta,
                           g=gamma,
                           p=population,
                           integ_t=integ_time)


@app.route('/basic_plot', methods=['GET'])
def basic_plot():
    try:
        # check for user input
        beta = request.args['b']
        gamma = request.args['g']
        population = request.args['p']
        time = request.args['t']

        beta = float(beta)
        gamma = float(gamma)
        population = int(population)
        time = int(time)
        bytes_object = run_simple(beta, gamma, population, time)

        bytes_object = run_simple(beta, gamma, population, time)

        return send_file(bytes_object,
                         attachment_filename='plot.png',
                         mimetype='image/png'
                         )
    except ValueError:
        make_response(
            'Probably features are out of a reasonable range and a number hits infinity',
            400)


@app.route('/fitted')
def fitted():
    crawl_wiki_pl()  # update data => only if first request on a new day

    t = time.time()  # to prevent browser from caching images

    current_data, _ = load_data(frontend=True)  # load data to render

    return render_template('fitted.html',
                           time=t,
                           tables=[current_data.to_html(classes='data')],
                           titles=current_data.columns)


@app.route('/fitted_plot', methods=['GET'])
def fitted_plot():
    try:
        bytes_object = run_fit()

        return send_file(bytes_object,
                         attachment_filename='fitted_plot.png',
                         mimetype='image/png'
                         )
    except ValueError:
        make_response(
            'Probably features are out of a reasonable range and a number hits infinity',
            400)


@app.route('/about')
def about():
    return render_template('about.html', title='About')


if __name__ == '__main__':
    load_dotenv(dotenv_path='config/.env')
    print(os.getenv("FLASK_APP"))
    print(os.getenv("FLASK_DEBUG"))
    app.run(debug=False)
