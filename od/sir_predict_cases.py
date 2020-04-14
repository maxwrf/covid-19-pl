import numpy as np
import pandas as pd
from scipy import optimize, integrate
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter('ignore')


class SIR():
    def __init__(self):
        self.load_data()

        self.y_infected = self.data['I']
        self.x_data = self.data.index

        self.N = self.population
        self.inf0 = self.y_infected[0]
        self.sus0 = self.N - self.inf0
        self.rec0 = 0.0

    def sir_model(self, y, x, beta, gamma):
        sus = -beta * y[0] * y[1] / self.N
        rec = gamma * y[1]
        inf = -(sus + rec)
        return sus, inf, rec

    def load_data(self):
        data = pd.read_csv('data/pl_regions.csv')
        data = data[['Confirmed cases|Total', 'Recoveries|Total',
                     'Deaths|Total', 'Date (DGS report)']]
        self.population = 10.28 * 1e06
        data['S'] = self.population - data['Confirmed cases|Total']
        data['I'] = data['Confirmed cases|Total'] - \
            (data['Recoveries|Total'] + data['Deaths|Total'])
        data['R'] = data['Recoveries|Total'] + data['Deaths|Total']
        self.data = data[['S', 'I', 'R']]

    def fit_odeint(self, x, beta, gamma, optimize=True):
        SIR = integrate.odeint(self.sir_model,
                               (self.sus0, self.inf0, self.rec0),
                               x,
                               args=(beta, gamma))
        if optimize:
            return SIR[:, 1]
        else:
            return SIR

    def run(self):
        popt, pcov = optimize.curve_fit(
            self.fit_odeint, self.x_data, self.y_infected)

        fitted_beta, fitted_gamma, _ = popt
        self.predicted = self.fit_odeint(
            self.x_data, beta=fitted_beta, gamma=fitted_gamma, optimize=False)
        print("Optimal parameters: beta =", fitted_beta,
              " and gamma = ", fitted_gamma)

    def plot(self):
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

        ax2 = ax.twinx()

        #ax2.plot(self.data['S'], label='S (Actuals)')
        ax.plot(self.data['I'], label='I (Actuals)')
        #ax.plot(self.data['R'], label='R (Actuals)')

        #ax2.plot(self.predicted[:, 0], label='S (Fitted)')
        ax.plot(self.predicted[:, 1], label='I (Fitted)')
        #ax.plot(self.predicted[:, 2], label='R (Fitted)')
        ax.legend()
        plt.show()


if __name__ == '__main__':
    model = SIR()
    model.run()
    model.plot()
