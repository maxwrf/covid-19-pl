import numpy as np
from abc import abstractmethod
from tqdm import tqdm
import matplotlib.pyplot as plt


class Solver():
    def __init__(self, f):
        self.f = f

    @abstractmethod
    def advance(self):
        """ advace differential one step"""
        raise NotImplementedError

    def set_initial_conditions(self, U0):
        if isinstance(U0, (int, float)):
            self.number_of_equaitions = 1
            U0 = float(U0)
        else:
            # System of equiations
            U0 = np.asarray(U0)
            self.number_of_equaitions = U0.size
        self.U0 = U0

    def solve(self, time_points):
        """
        Move forward by solving the system
        of equations at each point in time.
        Store the results in u.
        """
        self.t = np.asarray(time_points)
        n = self.t.size
        self.u = np.zeros((n, self.number_of_equaitions))

        self.u[0, :] = self.U0

        # integrate
        for i in tqdm(range(n - 1)):
            self.i = i
            self.u[i + 1] = self.advance()
        # TODO: same as n?
        return self.u[:i+2], self.t[:i+2]


class ForwardEuler(Solver):
    def advance(self):
        u, f, i, t = self.u, self.f, self.i, self.t
        dt = t[i + 1] - t[i]
        return u[i, :] + dt * f(u[i, :], t[i])


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


def beta(t):
    """
    A change in beta will exactly model different government
    interventions => t could change over time
    """
    # return 0.0005 if t <= 10 else 0.0001
    return 0.0005


if __name__ == '__main__':
    sir = SIR(beta, 0.1, 1500, 1, 0)
    solver = ForwardEuler(sir)
    solver.set_initial_conditions(sir.initial_conds)

    # Looking at a 60 days time span using 1001 integration time steps
    time_steps = np.linspace(0, 60, 1001)

    # Run solver
    u, t = solver.solve(time_steps)

    Spreds, Ipreds, Rpreds = u[:, 0], u[:, 1], u[:, 2]

    fig, ax = plt.subplots(1, 1)

    ax.plot(Spreds)
    ax.plot(Ipreds)
    ax.plot(Rpreds)

    plt.show()
