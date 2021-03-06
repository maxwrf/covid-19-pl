from source.sir_model.sir_model import SIR
from source.solver.forward_euler import ForwardEuler
import numpy as np
from source.solver.runge_kutta4 import RK4
import pandas as pd
from source.scraper.wiki_pl_scraper import crawl_wiki_pl
import os


def load_data(frontend=False):
    # TODO: move somewhere else
    population = 10.28 * 1e06
    try:
        data = pd.read_csv(os.getcwd() + '/data/pl_regions.csv')
    except BaseException:
        crawl_wiki_pl()
        data = pd.read_csv(os.getcwd() + '/data/pl_regions.csv')

    data = data.iloc[:-3, :]  # last rows are total
    data = data[['Confirmed cases|Total', 'Recoveries|Total',
                 'Deaths|Total', 'Date (DGS report)']]
    data[['Confirmed cases|Total', 'Recoveries|Total', 'Deaths|Total']] = data[[
        'Confirmed cases|Total', 'Recoveries|Total', 'Deaths|Total']].astype(int)
    data['S'] = population - data[['Confirmed cases|Total']]
    data['I'] = data['Confirmed cases|Total'] - \
        (data['Recoveries|Total'] + data['Deaths|Total'])
    data['R'] = data['Recoveries|Total'] + data['Deaths|Total']

    if frontend:
        data = data[['Date (DGS report)', 'S', 'I', 'R']]
        data.rename(columns={'S': 'Susceptible',
                             'I': 'Infectious',
                             'R': 'Recovered'}, inplace=True)
    else:
        data = data[['S', 'I', 'R']]

    return data, population


def run_simple(b, g, p, t):
    sir = SIR(b, g, p, 1, 0)
    print(p)
    # solver = ForwardEuler(sir)
    solver = RK4(sir)
    solver.set_initial_conditions(sir.initial_conds)

    # Looking at a t (60) days time span using 1001 integration time steps
    time_steps = np.linspace(0, t, 1001)

    # Run solver
    u, t = solver.solve(time_steps)
    Spreds, Ipreds, Rpreds = u[:, 0], u[:, 1], u[:, 2]

    bytes_object = SIR.sir_plot(Spreds, Ipreds, Rpreds, t)

    return bytes_object


def fit():
    beta_fitted, gamma_fitted = SIR.fit(*load_data())
    return beta_fitted, gamma_fitted


def plot_fit(beta_fitted, gamma_fitted):
    data, population = load_data()
    S0, I0, R0 = [
        n/population for n in (data['S'].iloc[0],
                               data['I'].iloc[0],
                               data['R'].iloc[0],)]
    print(beta_fitted, gamma_fitted)
    sir = SIR(beta_fitted, gamma_fitted, S0, I0, R0)
    solver = ForwardEuler(sir)
    solver.set_initial_conditions(sir.initial_conds)
    time_steps = np.linspace(0, 60, 1001)
    u, t = solver.solve(time_steps)
    Spreds, Ipreds, Rpreds = u[:, 0] * \
        population, u[:, 1]*population, u[:, 2]*population
    bytes_object = SIR.sir_plot(Spreds, Ipreds, Rpreds, t)

    return bytes_object


if __name__ == '__main__':
    run_simple()
