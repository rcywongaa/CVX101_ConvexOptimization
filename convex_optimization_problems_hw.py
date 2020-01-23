import cvxpy as cp
import numpy as np
import pdb

n=20
pbar = np.ones((n, 1))*.03 + np.concatenate([np.random.rand(n-1, 1), [[0.]]])*0.12
S = np.random.randn(n,n)
S = np.transpose(S)@S
S = S/max(abs(np.diagonal(S)))*.2
S[:,n-1] = np.zeros(n)
S[n-1,:] = np.zeros(n)
x_unif = np.ones((n,1))/n

def risk(x):
    x = np.array(x)
    return np.sqrt(x.T.dot(S.dot(x)))[0][0]

def expected_return(x):
    x = np.array(x)
    return pbar.T.dot(x)[0][0]

def print_portfolio(x, name):
    x = np.array(x)
    print(str(name) + ": expected = " + str(expected_return(x)) + ", risk = " + str(risk(x)))

print_portfolio(x_unif, "uniform")

required_return = expected_return(x_unif)

x = cp.Variable((n,1))

sum_to_one_constraint = np.ones((n,1)).T@x == 1
required_return_constraint = pbar.T@x == required_return

objective = cp.Minimize(cp.quad_form(x, 2*S))

min_risk_prob = cp.Problem(objective, [sum_to_one_constraint, required_return_constraint])
min_risk_prob.solve()
print("Status = " + str(min_risk_prob.status))
print_portfolio(x.value, "min risk")

long_only_constraint = x >= 0
long_only_prob = cp.Problem(objective, [
    sum_to_one_constraint,
    required_return_constraint,
    long_only_constraint])
long_only_prob.solve()
print("Status = " + str(long_only_prob.status))
print_portfolio(x.value, "long only")

limit_short_constraint = np.ones((n,1)).T@cp.neg(x) <= 0.5
limit_short_prob = cp.Problem(objective, [
    sum_to_one_constraint,
    required_return_constraint,
    limit_short_constraint])
limit_short_prob.solve()
print("Status = " + str(limit_short_prob.status))
print_portfolio(x.value, "limit short")
