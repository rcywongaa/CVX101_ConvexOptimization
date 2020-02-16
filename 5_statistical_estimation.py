import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pdb

def Maximum_likelihood_estimation_of_an_increasing_nonnegative_signal():
    '''
    Setup
    '''
    N = 100
    xtrue = np.zeros((N,1))
    xtrue[0:40] = 0.1
    xtrue[49] = 2 # matlab is 1-indexed
    xtrue[70:80] = 0.15
    xtrue[79] = 1 # matlab is 1-indexed
    xtrue = np.cumsum(xtrue, axis=0)

    h = np.array([[1, -0.85, 0.7, -0.3]])
    k = len(h)
    yhat = np.array([np.convolve(h.flatten(), xtrue.flatten())]).T
    y = yhat[:-3] + np.array([
        [-0.43],[-1.7],[0.13],[0.29],[-1.1],[1.2],[1.2],[-0.038],[0.33],[0.17],
        [-0.19],[0.73],[-0.59],[2.2],[-0.14],[0.11],[1.1],[0.059],[-0.096],[-0.83],
        [0.29],[-1.3],[0.71],[1.6],[-0.69],[0.86],[1.3],[-1.6],[-1.4],[0.57],
        [-0.4],[0.69],[0.82],[0.71],[1.3],[0.67],[1.2],[-1.2],[-0.02],[-0.16],
        [-1.6],[0.26],[-1.1],[1.4],[-0.81],[0.53],[0.22],[-0.92],[-2.2],[-0.059],
        [-1],[0.61],[0.51],[1.7],[0.59],[-0.64],[0.38],[-1],[-0.02],[-0.048],
        [4.3e-05],[-0.32],[1.1],[-1.9],[0.43],[0.9],[0.73],[0.58],[0.04],[0.68],
        [0.57],[-0.26],[-0.38],[-0.3],[-1.5],[-0.23],[0.12],[0.31],[1.4],[-0.35],
        [0.62],[0.8],[0.94],[-0.99],[0.21],[0.24],[-1],[-0.74],[1.1],[-0.13],
        [0.39],[0.088],[-0.64],[-0.56],[0.44],[-0.95],[0.78],[0.57],[-0.82],[-0.27]])
    plt.plot(range(len(xtrue)), xtrue, label="xtrue")
    plt.plot(range(len(y)), y, label="y")

    x_ml = cp.Variable((N,1))
    constraints = [x_ml >= 0] + [x_ml[i+1] >= x_ml[i] for i in range(N-1)]
    # obj = cp.sum(cp.log(y - cp.conv(h, x_ml)))
    # prob = cp.Problem(cp.Maximize(obj), constraints)
    obj = cp.norm2(y - cp.conv(h, x_ml))
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve()
    print("Status = " + str(prob.status))
    plt.plot(range(len(x_ml.value)), x_ml.value, label="x_ml")

    x_ml_free = cp.Variable((N,1))
    obj = cp.norm2(y - cp.conv(h, x_ml_free))
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve()
    print("Status = " + str(prob.status))
    plt.plot(range(len(x_ml_free.value)), x_ml_free.value, label="x_ml_free")

    plt.legend()
    plt.show()

def Worst_case_probability_of_loss():
    mu = [8., 20.]
    sigma = [6., 17.5]
    rho = -0.25
    n = 100
    prob_R = cp.Variable((n, n))
    r = np.linspace(-30., 70., n, endpoint=True)

    def marginal(r_i, k):
        return np.exp(-(r_i-mu[k])**2/(2*sigma[k]**2)) / np.sum([np.exp(-(r_j - mu[k])**2/(2*sigma[k]**2)) for r_j in r])

    nonnegative_constraint = prob_R >= 0.
    sum_to_one_constraint = cp.sum(prob_R) == 1.0
    r1_marginal_distribution_constraint = [cp.sum(prob_R[i,:]) == marginal(r[i], 0) for i in range(n)]
    r2_marginal_distribution_constraint = [cp.sum(prob_R[:,i]) == marginal(r[i], 1) for i in range(n)]

    covariance = 0.0
    for r1_idx in range(n):
        for r2_idx in range(n):
            covariance += prob_R[r1_idx, r2_idx]*(r[r1_idx]-mu[0])*(r[r2_idx]-mu[1])
    correlation_constraint = covariance == rho*sigma[0]*sigma[1]
    p_loss = 0.0

    for r1_idx in range(n):
        for r2_idx in range(n):
            if r[r1_idx] + r[r2_idx] <= 0:
                p_loss += prob_R[r1_idx, r2_idx]

    prob = cp.Problem(cp.Maximize(p_loss), (
        [nonnegative_constraint]
        + [sum_to_one_constraint]
        + r1_marginal_distribution_constraint
        + r2_marginal_distribution_constraint
        + [correlation_constraint]))

    # prob.solve(solver="SCS")
    prob.solve()
    print("Status: " + str(prob.status))
    print("p_loss = " + str(p_loss.value))
    r1, r2 = np.meshgrid(r, r)
    fig = plt.figure()
    ax1 = fig.add_subplot(211, projection='3d')
    ax1.plot_surface(r1, r2, prob_R.value, cmap=cm.coolwarm)
    ax2 = fig.add_subplot(212, projection='3d')
    ax2.contour(r1, r2, prob_R.value, np.linspace(0.001, 0.020, 5, endpoint=True), cmap=cm.coolwarm)
    plt.show()

# Maximum_likelihood_estimation_of_an_increasing_nonnegative_signal()
Worst_case_probability_of_loss()
