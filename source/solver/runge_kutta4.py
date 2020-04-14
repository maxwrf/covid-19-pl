from solver.base_solver import Solver


class RK4(Solver):
    def advance(self):
        u, f, i, t = self.u, self.f, self.i, self.t
        dt = t[i + 1] - t[i]
        yi = u[i, :]
        ti = t[i]
        k1 = f(yi, ti)
        k2 = f(yi + k1 * dt / 2, ti + dt / 2)
        k3 = f(yi + k2 * dt / 2, ti + dt / 2)
        k4 = f(yi + k3 * dt, ti + dt)

        return u[i, :] + dt / 6 * (k1 + 2 * k2 + 2*k3 + k4)
