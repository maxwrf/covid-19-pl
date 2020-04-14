import numpy as np
import matplotlib.pyplot as plt


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
