import cvxpy as cp
import numpy as np
import pdb
import matplotlib.pyplot as plt

def Numerical_perturbation_analysis_example():
    x = cp.Variable(2)
    u = cp.Parameter(2)

    obj = cp.Minimize(cp.quad_form(x, np.array(
        [[1., -0.5],
        [-0.5, 2.]]))
        - x[0])

    u.value = [-2, -3]

    constraints = [
            x[0] + 2*x[1] <= u[0],
            x[0] - 4*x[1] <= u[1],
            x[0] + x[1] >= -5]
    prob = cp.Problem(obj, constraints)

    prob.solve()
    p_star = obj.value
    print("Status: " + str(prob.status))
    print("x = " + str(x.value))
    print("obj = " + str(p_star))

    lambda1 = constraints[0].dual_value
    lambda2 = constraints[1].dual_value
    lambda3 = constraints[2].dual_value
    print("lambda1 = " + str(lambda1))
    print("lambda2 = " + str(lambda2))
    print("lambda3 = " + str(lambda3))

    def d_obj(x):
        return np.array([2*x[0]-x[1]-1, 4*x[1]-x[0]])
    d_c1 = np.array([1., 2.])
    d_c2 = np.array([1., -4.])
    d_c3 = np.array([-1., -1.])

    KKT_condition = d_obj(x.value) + lambda1*d_c1 + lambda2*d_c2 + lambda3*d_c3
    print("KKT = " + str(KKT_condition))

    for sigma1 in [0.0, -0.1, 0.1]:
        for sigma2 in [0.0, -0.1, 0.1]:
            print("sigma1 = " + str(sigma1) + ", sigma2 = " + str(sigma2))
            p_pred = p_star - lambda1*sigma1 - lambda2*sigma2
            u.value = [-2 + sigma1, -3 + sigma2]
            prob.solve()
            p_exact = obj.value
            print("p_pred = " + str(p_pred) + ", p_exact = " + str(p_exact) + ", p_exact - p_pred = " + str(p_exact - p_pred))

def A_simple_example():

    ### (a)
    x = cp.Variable(1)

    obj = cp.Minimize(cp.quad_form(x, np.array([[1]])) + 1)
    constraints = [cp.quad_form(x, np.array([[1]])) -6*x + 8 <= 0]

    prob = cp.Problem(obj, constraints)
    prob.solve()
    p_star = obj.value
    print("Status: " + str(prob.status))
    print("x = " + str(x.value))
    print("obj = " + str(p_star))

    ### (b)

    # Plot objective
    obj_func = lambda x : x**2 + 1
    x_vals = np.arange(-1.0, 5.0, 0.1)
    y_vals = [obj_func(x) for x in x_vals]
    plt.plot(x_vals, y_vals)

    # Plot feasible set
    feasible_x_vals = np.arange(2., 4., 0.1)
    plt.fill_between(feasible_x_vals, [obj_func(x) for x in feasible_x_vals], 20)

    # Plot optimal point and value
    plt.plot(x.value, p_star, 'ro')

    # Plot Lagrangian
    lagrangian_func = lambda x, l : x**2 + 1 + l*(x-2)*(x-4)
    for lamb in [1., 2., 3.]:
        lamb_y_vals = [lagrangian_func(x, lamb) for x in x_vals]
        plt.plot(x_vals, lamb_y_vals, color=np.random.rand(3), label="l=" + str(lamb))

    # Show plot
    plt.legend()
    plt.show()

    ### (c)

    '''
    Derive Lagrange dual function
    (1+l)x**2 - 6l*x + 8l +1
    Differentiate with respect to x and set equal 0
    2(1+l)x -6l = 0
    x = 3l/(1+l)
    Substitute back to original Lagrange dual function
    (1+l)(3l/(1+l))**2 - 6l*(3l/(1+l) + 8l + 1
    = 9l**2/(1+l) - 18l**2/(1+l) + 8l + 1
    = -9l**2/(1+l) + 8l + 1
    '''

    l = cp.Variable(1)
    dual_obj = cp.Maximize(-9*cp.quad_over_lin(l, 1+l) + 8*l + 1)
    dual_prob = cp.Problem(
            dual_obj,
            [l >= 0])
    dual_prob.solve()
    print("l_star = " + str(l.value))
    # l_star = 2

    ### (d)

    '''
    Solving the quadratic equation: x**2 - 6x + 8 - u = 0
    x = (6 +- sqrt(6**2 - 4*(8-u)))/2
      = (6 +- sqrt(36-32 + 4u))/2
      = (6 +- 2*sqrt(1+u))/2
      = 3 +- sqrt(1+u)
    From previous results, we know
    x_star = 3 - sqrt(1+u)

    p_star = x_star**2 + 1
           = 9 - 2*3*sqrt(1+u) + 1 + u + 1
           = 11 + u - 6*sqrt(1+u)
    '''

def Lagrangian_relaxation_of_Boolean_LP():
    ### (a)
    '''
    Lagrange dual function:
    minimize_x(cTx + (Ax-b)T*lambda1 + sum{i=1..n}(lambda2*xi(1-xi)))
    = minimize_x(cTx + (Ax)T*lambda1 -bT*lambda1 + sum{i=1..n}(lambda2*xi(1-xi)))
    = -bT*lambda1 + minimize_x(cTx + (Ax)T*lambda1 + sum{i=1..n}(lambda2*xi(1-xi)))
    = -bT*lambda1 + minimize_x((cT + AT*lambda1)x + sum{i=1..n}(lambda2*xi(1-xi)))
    = -bT*lambda1 + minimize_x(sum{i=1..n}((ci + ai.T*lambda1)xi + lambda2*(-xi**2 + xi)))

    In order for minimize_x(sum{i=1..n}((ci + ai.T*lambda1)xi + lambda2*(-xi**2 + xi))) to be bounded,
    lambda2 must be <=0,
    or
    lambda2*(-xi**2 + xi) = lambda2'*(xi**2 - xi) for lambda2' >= 0

    The minimization becomes
    lambda2'*xi**2 + (ci + ai.T*lambda1 - lambda2')xi

    Taking derivative and setting to 0, 2*lambda2'*xi + ci + ai.T*lambda1 - lambda2' = 0
    xi = -(ci + ai.T*lambda1 - lambda2')/(2*lambda2')

    Substituting back in, the minimization becomes
    1/4*(ci + ai.T*lambda1 - lambda2')**2/lambda2' - (ci + ai.T*lambda1 - lambda2')**2/(2*lambda2')
    = -1/4*(ci + ai.T*lambda1 - lambda2')**2 / lambda2'

    The lagrange dual function = -bT*lambda1 - sum{i=1..n}(1/4*(ci + ai.T*lambda1 - lambda2')**2 / lambda2')

    We can maximize lambda2' analytically to simplify the dual
    Taking derivative of -1/4*(ci + ai.T*lambda1 - lambda2')**2 / lambda2' w.r.t. lambda2' and setting 0
    1/2*(ci + ai.T*lambda1 - lambda2')/lambda2 + 1/4*(ci + ai.T*lambda1 - lambda2')**2 / (lambda2'**2) = 0
    1 + 1/2*(ci + ai.T*lambda1 - lambda2')/lambda2' = 0 or 1/2*(ci + ai.T*lambda1 - lambda2')/lambda2 = 0
    ci + ai.T*lambda1 = -lambda2                        or ci + ai.T*lambda1 - lambda2' = 0
    Max of -1/4*(ci + ai.T*lambda1 - lambda2')**2 / lambda2' 
    = max(0, -(ci + ai.T*lambda1)
    = min(0, ci + ai.T*lambda1)

    The lagrange dual function = -bT*lambda1 + sum{i=1..n} min(0, ci + ai.T*lambda1)
    '''

    ### (b)
    '''
    LP Relaxation dual: minimize cTx + (Ax-b)T*lambda1 + sum{i=1..n}(-lambda2*xi + lambda3(xi - 1))
    = minimize (cT + AT*lambda1)x - bT*lambda1 - lambda2.T*x + lambda3.T*x - lambda3.T*1
    = minimize -bT*lambda1 - lamda3.T*1 + (c + A*lambda1 - lambda2 + lambda3)Tx
    '''
    pass

Numerical_perturbation_analysis_example()
A_simple_example()
Lagrangian_relaxation_of_Boolean_LP()
