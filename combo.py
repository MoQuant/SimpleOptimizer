tickers = ['SPY','AAPL','GS','WMT','JNJ','NVDA','AMZN','NFLX','TSLA']

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Beta(x, y):
    coef = np.cov(x, y)
    beat = coef[0][1]
    box = coef[0][0]
    return beat / box

def Capital_Asset_Pricing_Model(rf, rm, beta):
    return rf + beta*(rm - rf)

datasets = {tick:pd.read_csv(f'{tick}.csv') for tick in tickers}
close = np.array([datasets[tick]['adjClose'].values.tolist() for tick in tickers]).T

rate_of_return = close[:-1]/close[1:] - 1

X = rate_of_return.T
X0 = X[0]

market_rate = close.T[0][0] / close.T[0][-1] - 1
risk_free_rate = 0.055

BETA, CAPM = [], []

for x in X:
    beta = Beta(X0, x)
    capm = Capital_Asset_Pricing_Model(risk_free_rate, market_rate, beta)
    BETA.append(beta)
    CAPM.append(capm)


from scipy.optimize import minimize

def MinimizeRisk(beta, capm, target_return):
    beta, capm = np.array(beta), np.array(capm)
    def objective(W):
        return W @ beta
    def constraint(W):
        return W @ capm - target_return
    def constraint2(W):
        return W @ np.ones(len(W)) - 1
    def constraint3(W):
        return W
    W = np.ones(len(beta))
    cons = [{'type':'ineq','fun':constraint},
            {'type':'eq','fun':constraint2},
            {'type':'ineq','fun':constraint3}]
    res = minimize(objective, W, method='SLSQP', bounds=None, constraints=cons)
    return res.x

def MaximizeReturn(beta, capm, target_beta):
    beta, capm = np.array(beta), np.array(capm)
    def objective(W):
        return -(W @ capm)
    def constraint(W):
        return -(W @ beta - target_beta)
    def constraint2(W):
        return W @ np.ones(len(W)) - 1
    def constraint3(W):
        return W
    W = np.ones(len(beta))
    cons = [{'type':'ineq','fun':constraint},
            {'type':'eq','fun':constraint2},
            {'type':'ineq','fun':constraint3}]
    res = minimize(objective, W, method='SLSQP', bounds=None, constraints=cons)
    return res.x


minrisk, maxretn = MinimizeRisk(BETA, CAPM, 0.13), MaximizeReturn(BETA, CAPM, 1.4)

fig = plt.figure(figsize=(13, 8))
ax = fig.add_subplot(121)
ay = fig.add_subplot(122)

target_rtn = 0.13
target_beta = 1.25

dot = lambda x, y: np.sum([i*j for i, j in zip(x, y)])

minimized_risk = dot(minrisk, BETA)
maximized_return = dot(maxretn, CAPM)

titleA = f'Minimized Risk Portfolio\nTargetCAPM: {target_rtn}\nPortfolio Beta: {round(minimized_risk, 3)}'
titleB = f'Maximized Return Portfolio\nTargetBeta: {target_beta}\nPortfolio Return: {round(maximized_return, 3)}'


ax.set_title(titleA)
ay.set_title(titleB)

s_tickers = [f'{i} | Beta: {round(j, 3)}' for i, j in zip(tickers, BETA)]

ax.pie(minrisk, labels=s_tickers, autopct='%1.1f%%')
ay.pie(maxretn, labels=s_tickers, autopct='%1.1f%%')



plt.show()













    
