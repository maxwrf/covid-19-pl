import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error


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
    cut = int(0.8 * len(data))
    train = data.iloc[:cut, :]
    test = data.iloc[cut:, :]
    return train, test, population


def dS_dt(beta, S, I):
    return -beta * S * I


def dI_dt(beta, S, I, gamma):
    return beta * S * I - gamma * I


def dR_dt(gamma, I):
    return gamma * I


def SIR_model(t, y, beta=3.6, gamma=2.9):
    S, I, R = y

    # if callable(beta):
    #     pass

    # if callable(gamma):
    #     pass

    S_out = dS_dt(beta, S, I)
    I_out = dI_dt(beta, S, I, gamma)
    R_out = dR_dt(gamma, I)
    return [S_out, I_out, R_out]


def predict():
    N = 100000  # Population size
    n_infected = 1
    max_days = 100
    initial_state = [(N - n_infected) / N, n_infected / N, 0]
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
    fig.suptitle('SIR model')

    ax.plot(S, label='Susceptible')
    ax.plot(I, label='Infected')
    ax.plot(R, label='Recovered')

    ax.legend()
    plt.show()


def plot_train(prediction, data):
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    fig.suptitle('SIR model')

    ax.plot(prediction['S'], label='Susceptible')
    ax.plot(prediction['I'], label='Infected')
    ax.plot(prediction['R'], label='Recovered')

    ax.scatter(data.index, data['S'], label='Susceptible Actual')
    ax.scatter(data.index, data['I'], label='Infected Actual')
    ax.scatter(data.index, data['R'], label='Recovered Actual')

    ax.legend()
    plt.show()


def loss(params, data, population, return_solution=False,
         forecast_days=0, optim_days=20):
    beta, gamma = params
    N = population
    n_infected = data['I'].iloc[0]
    max_days = len(data) + forecast_days

    initial_state = [(N - n_infected) / N, n_infected / N, 0]
    args = (beta, gamma)

    prediction = solve_ivp(
        SIR_model, [0, max_days], initial_state,
        args=args, t_eval=np.arange(max_days))

    S, I, R = prediction.y

    y_pred_infected = np.clip(I, 0, np.inf) * population
    y_true_cases = data['I'].values
    y_pred_recovered = np.clip(R, 0, np.inf) * population
    y_true_recovered = data['R'].values

    optim_days = int(min(optim_days, len(data)))

    weights = 1 / np.arange(1, optim_days+1)[::-1]

    mse_infected = mean_squared_log_error(
        y_true_cases[-optim_days:], y_pred_infected[-optim_days:], weights)
    mse_recovered = mean_squared_log_error(
        y_true_recovered[-optim_days:], y_pred_recovered[-optim_days:],
        weights)

    mse = np.mean([mse_infected])

    if return_solution:
        return mse, prediction
    return mse


def fit(initial_guess=[0.7, 0.2], bounds=[(0.02, 2), (0.0001, 0.3)]):
    train, test, population = load_data()

    res = minimize(loss, initial_guess, bounds=bounds,
                   args=(train, population), method='L-BFGS-B')

    # how long should the forecast be?
    t = 100

    # compute SIR based on fitted X
    mse, sol = loss(res.x, train, population, True,
                    forecast_days=(len(test) + t))
    S, I, R = sol.y

    # gather predictions in data frane scaled by populatiobn size
    pred = pd.DataFrame(
        {
            'S': np.clip(S, 0, np.inf) * population,
            'I': np.clip(I, 0, np.inf) * population,
            'R': np.clip(R, 0, np.inf) * population
        }
    )

    pred_test = pred.iloc[len(train):(len(train) + len(test)), :]

    mse_infected = mean_squared_log_error(test['I'], pred_test['I'])
    mse_recovered = mean_squared_log_error(test['R'], pred_test['R'])

    mse = np.mean([mse_infected, mse_recovered])

    print(mse)
    print(res.x)
    print(res.x[0] / res.x[1])
    plot_train(pred, pd.concat([train, test]))
    return mse


if __name__ == '__main__':
    pred = predict()
    plot(pred)
    fit()
