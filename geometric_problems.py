import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import pdb

def Three_way_linear_classification():
    '''
    Start of sep3way_data.m
    '''
    M = 20
    N = 20
    P = 20

    X = np.array([[3.5674, 4.1253, 2.8535, 5.1892, 4.3273, 3.8133, 3.4117,
        3.8636, 5.0668, 3.9044, 4.2944, 4.7143, 3.3082, 5.2540,
        2.5590, 3.6001, 4.8156, 5.2902, 5.1908, 3.9802 ],
        [-2.9981, 0.5178, 2.1436, -0.0677, 0.3144, 1.3064, 3.9297,
        0.2051, 0.1067, -1.4982, -2.4051, 2.9224, 1.5444, -2.8687,
        1.0281, 1.2420, 1.2814, 1.2035, -2.1644, -0.2821]])

    Y = np.array([[-4.5665, -3.6904, -3.2881, -1.6491, -5.4731, -3.6170, -1.1876,
        -1.0539, -1.3915, -2.0312, -1.9999, -0.2480, -1.3149, -0.8305,
        -1.9355, -1.0898, -2.6040, -4.3602, -1.8105, 0.3096],
        [2.4117, 4.2642, 2.8460, 0.5250, 1.9053, 2.9831, 4.7079,
        0.9702, 0.3854, 1.9228, 1.4914, -0.9984, 3.4330, 2.9246,
        3.0833, 1.5910, 1.5266, 1.6256, 2.5037, 1.4384]])

    Z = np.array([[1.7451, 2.6345, 0.5937, -2.8217, 3.0304, 1.0917, -1.7793,
        1.2422, 2.1873, -2.3008, -3.3258, 2.7617, 0.9166, 0.0601,
        -2.6520, -3.3205, 4.1229, -3.4085, -3.1594, -0.7311],
        [-3.2010, -4.9921, -3.7621, -4.7420, -4.1315, -3.9120, -4.5596,
        -4.9499, -3.4310, -4.2656, -6.2023, -4.5186, -3.7659, -5.0039,
        -4.3744, -5.0559, -3.9443, -4.0412, -5.3493, -3.0465]])

    '''
    Beginning of solution code
    '''
    a1_var = cp.Variable((2,1))
    a2_var = cp.Variable((2,1))
    a3_var = cp.Variable((2,1))
    b1_var = cp.Variable((1,1))
    b2_var = cp.Variable((1,1))
    b3_var = cp.Variable((1,1))

    regularization = 1e-3

    f1 = lambda z : cp.matmul(a1_var.T, z) - b1_var
    f2 = lambda z : cp.matmul(a2_var.T, z) - b2_var
    f3 = lambda z : cp.matmul(a3_var.T, z) - b3_var

    obj1 = cp.sum(f1(X)) - regularization*(cp.norm2(a1_var) + cp.norm2(b1_var))
    obj2 = cp.sum(f2(Y)) - regularization*(cp.norm2(a2_var) + cp.norm2(b2_var))
    obj3 = cp.sum(f3(Z)) - regularization*(cp.norm2(a3_var) + cp.norm2(b3_var))

    constraints1 = [
            f1(X) >= f2(X),
            f1(X) >= f3(X)]
    constraints2 = [
            f2(Y) >= f1(Y),
            f2(Y) >= f3(Y)]
    constraints3 = [
            f3(Z) >= f1(Z),
            f3(Z) >= f2(Z)]

    prob = cp.Problem(cp.Maximize(obj1), constraints1 + constraints2 + constraints3)
    prob.solve()

    print("Problem: " + str(prob.status))
    print("Objective = " + str(prob.value))

    a1 = a1_var.value
    a2 = a2_var.value
    a3 = a3_var.value
    b1 = b1_var.value
    b2 = b2_var.value
    b3 = b3_var.value
    print("a1 = " + str(a1))
    print("a2 = " + str(a2))
    print("a3 = " + str(a3))
    print("b1 = " + str(b1))
    print("b2 = " + str(b2))
    print("b3 = " + str(b3))

    '''
    End of solution code
    '''

    # comment out the following line after filling in cvx part!
    # values below are not right!!
    # a1=np.array([[1.],[1.]])
    # a2=np.array([[1.],[-5.]])
    # a3=np.array([[-1.],[-1.]])
    # b1=np.array([[0.]])
    # b2=np.array([[0.]])
    # b3=np.array([[0.]])

    '''
    Numpy does not have Matlab's '\' operator (mldivide)
    Here is an implementation taken from:
    https://stackoverflow.com/questions/33559946/numpy-vs-mldivide-matlab-operator
    '''
    def matlab_backslash(A, b):
        from itertools import combinations
        num_vars = A.shape[1]
        rank = np.linalg.matrix_rank(A)
        if rank == num_vars:
            return np.linalg.lstsq(A, b, rcond=None)[0]    # not under-determined
        else:
            for nz in combinations(range(num_vars), rank):    # the variables not set to zero
                try:
                    sol = np.zeros((num_vars, 1))
                    sol[nz, :] = np.asarray(np.linalg.solve(A[:, nz], b))
                    return sol
                except np.linalg.LinAlgError:
                    pass                    # picked bad variables, can't solve
        raise Exception("Undefined")

    # now let's plot the three-way separation induced by
    # a1,a2,a3,b1,b2,b3
    # find maximally confusing point
    p = matlab_backslash(np.vstack([(a1-a2).T,(a1-a3).T]),(np.vstack([(b1-b2),(b1-b3)])))
    print("maximally confusing point: " + str(p))

    # plot
    t = np.arange(-7, 7, 0.01);
    u1 = a1-a2; u2 = a2-a3; u3 = a3-a1;
    v1 = b1-b2; v2 = b2-b3; v3 = b3-b1;
    line1 = ((-t*u1[0]+v1)/u1[1]).flatten()
    idx1 = np.flatnonzero(u2.T.dot(np.vstack([t,line1]))-v2>0)
    line2 = ((-t*u2[0]+v2)/u2[1]).flatten()
    idx2 = np.flatnonzero(u3.T.dot(np.vstack([t,line2]))-v3>0)
    line3 = ((-t*u3[0]+v3)/u3[1]).flatten()
    idx3 = np.flatnonzero(u1.T.dot(np.vstack([t,line3]))-v1>0)

    plt.plot(X[0],X[1],'*',Y[0],Y[1],'ro',Z[0],Z[1],'g+',
            t[idx1],line1[idx1],'k',t[idx2],line2[idx2],'k',t[idx3],line3[idx3],'k');
    plt.axis([-7,7,-7,7]);
    plt.show()
    '''
    End of sep3way_data.m
    '''

def Fitting_a_sphere_to_data():
    '''
    sphere_fit_data.m
    '''
    U = np.array([
        [-3.8355737e+00,  5.9061250e+00],
        [-3.2269177e+00,  7.5112709e+00],
        [-1.6572955e+00,  7.4704730e+00],
        [-2.8202585e+00,  7.7378120e+00],
        [-1.7831869e+00,  5.4818448e+00],
        [-2.1605783e+00,  7.7231450e+00],
        [-2.0960803e+00,  7.7072529e+00],
        [-1.3866295e+00,  6.1452654e+00],
        [-3.2077849e+00,  7.6023307e+00],
        [-2.0095986e+00,  7.6382459e+00],
        [-2.0965432e+00,  5.2421510e+00],
        [-2.8128775e+00,  5.1622157e+00],
        [-3.6501826e+00,  7.2585500e+00],
        [-2.1638414e+00,  7.6899057e+00],
        [-1.7274710e+00,  5.4564872e+00],
        [-1.5743230e+00,  7.3510769e+00],
        [-1.3761806e+00,  6.9730981e+00],
        [-1.3602495e+00,  6.9056362e+00],
        [-1.5257654e+00,  5.7518622e+00],
        [-1.9231176e+00,  7.6775030e+00],
        [-2.9296195e+00,  7.7561481e+00],
        [-3.2828270e+00,  5.4188036e+00],
        [-2.9078414e+00,  5.1741322e+00],
        [-3.5423007e+00,  5.5660735e+00],
        [-3.1388035e+00,  7.7008514e+00],
        [-1.7957226e+00,  5.4273243e+00],
        [-2.6267585e+00,  7.7336173e+00],
        [-3.6652627e+00,  7.2686635e+00],
        [-3.7394118e+00,  6.0293335e+00],
        [-3.7898021e+00,  5.9057623e+00],
        [-3.6200108e+00,  5.7754097e+00],
        [-3.0386294e+00,  5.3028798e+00],
        [-2.0320023e+00,  5.2594588e+00],
        [-2.9577808e+00,  5.3040353e+00],
        [-2.9146706e+00,  7.7731243e+00],
        [-3.2243786e+00,  5.4402982e+00],
        [-2.1781976e+00,  7.7681141e+00],
        [-2.2545150e+00,  5.2233652e+00],
        [-1.2559218e+00,  6.2741755e+00],
        [-1.8875105e+00,  5.4133273e+00],
        [-3.6122685e+00,  7.2743342e+00],
        [-2.6552417e+00,  7.7564498e+00],
        [-1.4127560e+00,  6.0732284e+00],
        [-3.7475311e+00,  7.2351834e+00],
        [-2.1367633e+00,  7.6955709e+00],
        [-3.9263527e+00,  6.2241593e+00],
        [-2.3118969e+00,  7.7636052e+00],
        [-1.4249518e+00,  7.1457752e+00],
        [-2.0196394e+00,  5.3154475e+00],
        [-1.4021445e+00,  5.9675466e+00]
        ]).T;
    ''''''
    N = U.shape[0]
    M = U.shape[1]

    def first_attempt():
        '''
        Calculate the minimum distance from center to all points
        '''
        constraints = []
        D = cp.Variable((M+1, M+1))
        constraints.append(D >= 0)
        constraints.append(D == D.T)
        for i in range(M):
            for j in range(M):
                if i == j:
                    constraints.append(D[i,j] == 0)
                else:
                    constraints.append(D[i,j] == np.sum((U[:,i] - U[:,j])**2))

        obj = cp.sum(D[M,:])
        constraints.append(D[M,M] == 0)

        one_perp = cp.Constant(np.identity(M+1) - 1./(M+1)*(np.ones((M+1, M+1))))
        constraints.append(one_perp@D@one_perp << 0.0) # D is negative semidefinite on 1 perp

        pdb.set_trace()
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve()
        print("Status = " + str(prob.status))

        D_opt = D.value
        d_ci = D_opt[M,:]

        '''
        Calculate minimum radius that satisfies the distances
        '''
        r_sq = cp.Variable()

        obj = cp.sum((d_ci - r_sq)**2)
        prob = cp.Problem(cp.Minimize(obj))
        prob.solve()

    '''
    Correct method: Least squares
    Express (||u_i - x_c||^2 - r^2)^2
    as (u_i^2 - 2*u_i*x_c + x_c^2 - r^2)^2
    Let t = x_c^2 - r^2
    we have (u_i^2 - 2*u_i*x_c + t)^2
    which is a QCQP in x_c and t
    with constraint that t <= x_c^2 (because r^2 >= 0) (this is not a quadratic constraint!)
    '''
    x_c = cp.Variable((2,1), name="x_c")
    t = cp.Variable(name="t")
    obj = cp.sum([cp.norm2(u_i.T@u_i - 2*u_i@x_c + t)**2 for u_i in U.T])
    # constraints = [cp.quad_form(x_c, np.identity(2)) >= t]
    constraints = []

    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve()
    print("Status = " + str(prob.status))
    print("t = " + str(t.value))
    print("x_c = " + str(x_c.value))

    x_c = x_c.value
    r_sq = x_c.T@x_c - float(t.value)
    r = np.sqrt(r_sq)
    print("r = " + str(r))

    plt.scatter(U[:][0], U[:][1])
    plt.gca().add_artist(plt.Circle(x_c, r, color='r', fill=False))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# Three_way_linear_classification()
Fitting_a_sphere_to_data()
