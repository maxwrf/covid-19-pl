from abc import abstractmethod
from tqdm import tqdm
import numpy as np


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
