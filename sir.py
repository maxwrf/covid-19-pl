import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def load_data():
    data = pd.read_csv('data/pl_regions.csv')
    data = data[['Confirmed cases|Total', 'Recoveries|Total',
                 'Deaths|Total', 'Date (DGS report)']]
    population = 10.28 * 1e06
    data['S'] = population - data['Confirmed cases|Total']
    data['I'] = data['Confirmed cases|Total'] - \
        (data['Recoveries|Total'] + data['Deaths|Total'])
    data['R'] = data['Recoveries|Total'] + data['Deaths|Total']
    data = data[['S', 'I', 'R']]


def dS_dt(beta, S, I):
    return -beta * S * I


def dI_dt(beta, S, I, gamma):
    return beta * S * I - gamma * I


def dR_dt(gamma, I):
    return gamma * I


def SIR_model(t, y, beta=3.6, gamma=2.9):
    S, I, R = y
    S_out = dS_dt(beta, S, I)
    I_out = dI_dt(beta, S, I, gamma)
    R_out = dR_dt(gamma, I)
    return [S_out, I_out, R_out]


def predict():
    N = 100000  # Population size
    n_infected = 1
    max_days = 100
    initial_state = [(N - n_infected) / N, n_infected / N, 0]
    R_t = 3.6  # reproductin number
    t_inf = 5.2  # average infectious period
    beta = 0.7
    gamma = 0.2
    args = beta, gamma
    prediction = solve_ivp(
        SIR_model, [0, max_days], initial_state,
        args=args, t_eval=np.arange(max_days))
    return prediction


def plot(prediction):
    S, I, R = prediction.y
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    ax.plot(S)
    ax.plot(I)
    ax.plot(R)

    plt.show()

# def loss(point, data, recovered, S0, I0, R0):
#     beta, gamma = point
#     span = len(data)
#     solution = solve_ivp(SIR_model, [0, span], [S0, I0, R0],
#                          t_eval=np.arange(0, span, 1), vectorized=True)
#     return np.sqrt(np.mean(solution.y[1] - data) ** 2)

# def train(data):
#     optimal = minimize(loss,
#                        [0.001, 0.001],
#                        args=(data['S'], data['R'], S0, I0, R0),
#                        method='L-BFGS-B',
#                        bounds=[(0.00000001, 0.4),
#                                (0.00000001, 0.4)])
#     print(optimal)
#     beta, gamma = optimal.x
#     return beta, gamma


if __name__ == '__main__':
    pred = predict()
    plot(pred)
