import json

import numpy as np
from numpy import mean, var
from scipy.stats import kurtosis, skew

r = 0.05
M = 10
N = 512
TN = [10, 50, 100, 252, 365, 512]
dt = 1 / N
Z = np.random.normal(0, 1, (N+1, M))
def mc(omega, alpha, beta, gamma, lambda_):
    num_point = N+1
    Rt = np.zeros((num_point, M))
    ht = np.zeros((num_point, M))
    ht[0] = (omega + alpha)/(1-beta-alpha*gamma**2)
    Rt[0] = 0
    for i in range(1, num_point):
        ht[i] = omega + beta*ht[i-1] + alpha*(Z[i-1] - gamma*np.sqrt(ht[i-1]))**2
        Rt[i] = r + lambda_*ht[i] + np.sqrt(ht[i])*Z[i]
    return Rt

Rt= mc(1e-6, 1.33e-6, 0.8, 5, 0.2)
print(Rt)

# index Rt from 0 to TN and get the four moments
for t in TN:
    Rt_t = Rt[:t+1]
    mean_t = mean(Rt_t)
    var_t = var(Rt_t)
    skew_t = skew(Rt_t)
    kurtosis_t = kurtosis(Rt_t)
    print(f"Mean at t={t}: {mean_t}")
    print(f"Variance at t={t}: {var_t}")
    print(f"Skewness at t={t}: {skew_t}")
    print(f"Kurtosis at t={t}: {kurtosis_t}")
