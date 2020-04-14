import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error
from scipy.optimize import minimize
from solver.runge_kutta4 import RK4


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
    def loss(params, data, population):
        beta_fitted, gamma_fitted = params

        # TODO: Set initial conditions to population or data!
        S0, I0, R0 = [
            n/population for n in (data['S'].iloc[0],
                                   data['I'].iloc[0],
                                   data['R'].iloc[0],)]

        sir = SIR(beta_fitted, gamma_fitted, S0, I0, R0)
        solver = RK4(sir)
        solver.set_initial_conditions(sir.initial_conds)

        optimization_days = len(data)

        time_steps = np.linspace(0, optimization_days, optimization_days)
        u, t = solver.solve(time_steps)
        Spreds, Ipreds, Rpreds = u[:, 0], u[:, 1], u[:, 2]

        weights = 1 / np.arange(1, optimization_days + 1)[::-1]

        msle_suspectibles = mean_squared_log_error(
            data['S'] / population, Spreds, weights)

        msle_infected = mean_squared_log_error(
            data['I'] / population, Ipreds, weights)

        msle_recovered = mean_squared_log_error(
            data['R'] / population, Rpreds, weights)

        msle = np.mean([msle_suspectibles,
                        msle_infected,
                        msle_recovered])
        print(msle)
        return msle

    @staticmethod
    def fit(data, population):
        initial_guess = [0.7, 0.2]
        bounds = [(0.02, 3),
                  (0.001, 2)]
        best_fit = minimize(SIR.loss,
                            initial_guess,
                            bounds=bounds,
                            args=(data, population),
                            method='L-BFGS-B')
        print(best_fit)
        beta, gamma = best_fit.x
        return beta, gamma
