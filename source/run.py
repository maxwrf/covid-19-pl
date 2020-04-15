from sir_model.sir_model import SIR
from solver.forward_euler import ForwardEuler
import numpy as np
from sir_model.dynamic_beta import beta
from sir_model.dynamic_gamma import gamma
from solver.runge_kutta4 import RK4
import pandas as pd


def load_data():
    # TODO: move somewhere else
    population = 10.28 * 1e06

    data = pd.read_csv('data/pl_regions.csv')
    data = data[['Confirmed cases|Total', 'Recoveries|Total',
                 'Deaths|Total', 'Date (DGS report)']]

    data['S'] = population - data['Confirmed cases|Total']
    data['I'] = data['Confirmed cases|Total'] - \
        (data['Recoveries|Total'] + data['Deaths|Total'])
    data['R'] = data['Recoveries|Total'] + data['Deaths|Total']
    data = data[['S', 'I', 'R']]

    return data, population


def run():
    sir = SIR(beta, gamma, 1500, 1, 0)
    # solver = ForwardEuler(sir)
    solver = RK4(sir)
    solver.set_initial_conditions(sir.initial_conds)

    # Looking at a 60 days time span using 1001 integration time steps
    time_steps = np.linspace(0, 60, 1001)

    # Run solver
    u, t = solver.solve(time_steps)
    Spreds, Ipreds, Rpreds = u[:, 0], u[:, 1], u[:, 2]
    SIR.sir_plot(Spreds, Ipreds, Rpreds, t)

    # fit Portugal data to generate beta and gamma
    # TODO: pass solver as argument
    beta_fitted, gamma_fitted = SIR.fit(*load_data())
    data, population = load_data()
    S0, I0, R0 = [
        n/population for n in (data['S'].iloc[0],
                               data['I'].iloc[0],
                               data['R'].iloc[0],)]
    sir = SIR(beta_fitted, gamma_fitted, S0, I0, R0)
    solver = ForwardEuler(sir)
    solver.set_initial_conditions(sir.initial_conds)
    time_steps = np.linspace(0, len(data) + 100, 10001)
    u, t = solver.solve(time_steps)
    Spreds, Ipreds, Rpreds = u[:, 0] * \
        population, u[:, 1]*population, u[:, 2]*population
    SIR.sir_plot(Spreds, Ipreds, Rpreds, t)


if __name__ == '__main__':
    run()
