import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error
from scipy.optimize import minimize
from source.solver.runge_kutta4 import RK4


class SIR():
    """
    dS/dT = -beta * S * I
    dI/dT = beta * S * I - gamma * I
    dR/dT = gamma * I
    """

    def __init__(self, beta, gamma, S0, I0, R0):
        """
        beta and gamma => System paramters
        S0, I0, R0 => Initial state
        """
        # Check for Time dependent beta and gamma
        if callable(beta):
            self.beta = beta
        else:
            # if not change to function for consistency
            self.beta = lambda t: beta

        if callable(gamma):
            self.gamma = gamma
        else:
            # if not change to function for consistency
            self.gamma = lambda t: gamma

        # store initial conditions
        self.initial_conds = [S0, I0, R0]

    def __call__(self, u, t):
        S, I, R = u
        return np.asarray([
            self.__dS_dt__(S, I, t),
            self.__dI_dt__(S, I, t),
            self.__dR_dt__(I, t)
        ])

    def __dS_dt__(self, S, I, t):
        return - self.beta(t) * S * I

    def __dI_dt__(self, S, I, t):
        return self.beta(t) * S * I - self.gamma(t) * I

    def __dR_dt__(self, I, t):
        return self.gamma(t) * I

    @staticmethod
    def sir_plot(S, I, R, t):
        if S.ndim == I.ndim == R.ndim == 1:
            fig, ax = plt.subplots(1, 1)
            ax.plot(S)
            ax.plot(I)
            ax.plot(R)

            plt.show()

    @staticmethod
    def loss(params, data, population, optim_days=20):
        # Unpack beta and gamma on which ti compute the loss
        beta_fitted, gamma_fitted = params

        # Get initial conditions from data
        S0, I0, R0 = [
            n/population for n in (data['S'].iloc[0],
                                   data['I'].iloc[0],
                                   data['R'].iloc[0],)]

        # Forward integrate to generate predictions
        sir = SIR(beta_fitted, gamma_fitted, S0, I0, R0)
        solver = RK4(sir)
        solver.set_initial_conditions(sir.initial_conds)
        time_steps = np.linspace(0, len(data), len(data))
        u, t = solver.solve(time_steps)
        Spreds, Ipreds, Rpreds = u[:, 0], u[:, 1], u[:, 2]

        # Compute loss given the predictions
        optim_days = int(min(optim_days, len(data)))
        weights = 1 / np.arange(1, optim_days + 1)[::-1]

        msle_suspectibles = mean_squared_log_error(
            data['S'][-optim_days:],
            np.clip(Spreds[-optim_days:], 0, np.inf) * population,
            weights)

        msle_infected = mean_squared_log_error(
            data['I'][-optim_days:],
            np.clip(Ipreds[-optim_days:], 0, np.inf) * population,
            weights)

        msle_recovered = mean_squared_log_error(
            data['R'][-optim_days:],
            np.clip(Rpreds[-optim_days:], 0, np.inf) * population,
            weights)

        msle = np.mean([
            # msle_suspectibles,
            msle_infected,
            # msle_recovered
        ])
        print(msle)
        return msle

    @staticmethod
    def fit(data, population):
        initial_guess = [.0001, .001]
        bounds = [(0.0002, .7),
                  (0.001, .7)]
        best_fit = minimize(SIR.loss,
                            initial_guess,
                            bounds=bounds,
                            args=(data, population),
                            method='L-BFGS-B')
        print(best_fit)
        beta, gamma = best_fit.x
        return beta, gamma
