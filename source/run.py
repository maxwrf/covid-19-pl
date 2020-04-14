from sir_model.sir_model import SIR
from solver.forward_euler import ForwardEuler
import numpy as np
from sir_model.dynamic_beta import beta
from sir_model.dynamic_gamma import gamma


def run():
    sir = SIR(beta, gamma, 1500, 1, 0)
    solver = ForwardEuler(sir)
    solver.set_initial_conditions(sir.initial_conds)

    # Looking at a 60 days time span using 1001 integration time steps
    time_steps = np.linspace(0, 60, 1001)

    # Run solver
    u, t = solver.solve(time_steps)
    Spreds, Ipreds, Rpreds = u[:, 0], u[:, 1], u[:, 2]
    SIR.sir_plot(Spreds, Ipreds, Rpreds, t)


if __name__ == '__main__':
    run()
