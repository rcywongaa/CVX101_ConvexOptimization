import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import pdb
import math

def Gradient_and_Newton_methods():
    n = 100
    m = 200
    A = np.random.random((m,n))

    '''
    Derivative of -sum{i=1..m}(log(1-a_i.T*x) - sum{i=1..n}(log(1-x_i^2))
    = -sum{i=1..m}(-a_ij/(1-a_i.T*x)) - (-2x_j/(1-x_j^2))
    '''
    def f(x):
        return -np.sum([np.log(1-A[i,:].T@x) for i in range(m)]) - np.sum([np.log(1-x[i]**2) for i in range(n)])

    def f_d(x):
        return np.array([
            -np.sum([-A[i,j]/(1-A[i,:].T@x) for i in range(m)]) - (-2*x[j])/(1 - x[j]**2) for j in range(n)])

    def backtracking_line_search(x, delta_x):
        alpha = 0.2
        beta = 0.5
        t = 1
        while math.isnan(f(x+t*delta_x)) or f(x+t*delta_x) > f(x) + alpha*t*f_d(x).T@delta_x:
            t = beta*t
        return t

    def part_a():
        x = np.zeros(n)
        nu = 1e-3

        f_vals = []
        iteration = 0
        while True:
            f_vals.append(f(x))
            print("Iter " + str(iteration) + ": " + str(f_vals[-1]))
            delta_x = -f_d(x)
            if (np.linalg.norm(delta_x) <= nu):
                break
            t = backtracking_line_search(x, delta_x)
            x += t*delta_x
            iteration += 1

        plt.plot(f_vals)
        plt.show()

    '''
    Using the property stated in section A.4.4 of the cvx book (p.645)
    -sum{i=1..m}(log(1-a_i.T*x) - sum{i=1..n}(log(1-x_i^2))
    = -sum{i=1..m}(log(1-a_i.T*x) - sum{i=1..n}(log(1-x_i) + log(1+x_i))
    can be seen as a composition with an affine function with
    g(x) = sum(log(x))
    g'(x) = 1/x
    g''(x) = diag(-1/x^2)
    Hence,
    f1(x) = sum{i=1..m}(log(1-a_i.T*x)
          = g(1 - A.T*x)
    f1''(x) = A.T*g''(1-A.T*x)*A

    f2(x) = sum{i=1..n}(log(1-x_i) + log(1+x_i))
          = g(1-x_i) + g(1+x_i)
    f2''(x) = g''(1-x) + g''(1+x)

    f''(x) = -f1''(x) - f2''(x)
    '''
    def f_dd(x):
        def g(x):
            return np.sum(np.log(x))
        def g_d(x):
            return 1/x
        def g_dd(x):
            return np.diag(-1/(x**2))
        f1_dd = A.T@g_dd(1 - A@x)@A
        f2_dd = g_dd(1-x) + g_dd(1+x)

        return -f1_dd - f2_dd

    def part_b():
        x = np.zeros(n)
        f_vals = []
        iteration = 0
        epsilon = 1e-8
        while True:
            f_vals.append(f(x))
            print("Iter " + str(iteration) + ": " + str(f_vals[-1]))
            f_dd_inv = np.linalg.inv(f_dd(x))
            delta_x = -f_dd_inv@f_d(x)
            lambda_2 = f_d(x).T@f_dd_inv@f_d(x)
            if lambda_2/2.0 <= epsilon:
                break
            t = backtracking_line_search(x, delta_x)
            x += t*delta_x
            iteration += 1

        plt.plot(f_vals)
        plt.show()

    part_a()
    part_b()

def Flux_balance_analysis_in_systems_biology():
    ''' fba_data.m '''
    # data file for flux balance analysis in systems biology
    # From Segre, Zucker et al "From annotated genomes to metabolic flux
    # models and kinetic parameter fitting" OMICS 7 (3), 301-316.

    # Stoichiometric matrix
    S = np.array([

        #M1    M2    M3    M4    M5    M6
        [ 1,    0,    0,    0,    0,    0 ],    #    R1:  extracellular -->  M1
        [-1,    1,    0,    0,    0,    0 ],    #    R2:  M1 -->  M2
        [-1,    0,    1,    0,    0,    0 ],    #    R3:  M1 -->  M3
        [ 0,   -1,    0,    2,   -1,    0 ],    #    R4:  M2 + M5 --> 2 M4
        [ 0,    0,    0,    0,    1,    0 ],    #    R5:  extracellular -->  M5
        [ 0,   -2,    1,    0,    0,    1 ],    #    R6:  2 M2 -->  M3 + M6
        [ 0,    0,   -1,    1,    0,    0 ],    #    R7:  M3 -->  M4
        [ 0,    0,    0,    0,    0,   -1 ],    #    R8:  M6 --> extracellular
        [ 0,    0,    0,   -1,    0,    0 ]     #    R9:  M4 --> cell biomass
        ]).T

    m,n = S.shape
    vmax = np.array([
        [10.10],  # R1:  extracellular -->  M1
        [100],    # R2:  M1 -->  M2
        [5.90],   # R3:  M1 -->  M3
        [100],    # R4:  M2 + M5 --> 2 M4
        [3.70],   # R5:  extracellular -->  M5
        [100],    # R6:  2 M2 --> M3 + M6
        [100],    # R7:  M3 -->  M4
        [100],    # R8:  M6 -->  extracellular
        [100]    # R9:  M4 -->  cell biomass
        ])
    ''''''

    def part_a():
        v = cp.Variable((n,1))
        obj = cp.Maximize(v[-1])
        constraints = [S@v == 0, v >= 0, v <= vmax]
        prob = cp.Problem(obj, constraints)
        prob.solve()
        print("G* = " + str(v.value[-1]))
        print("Dual S@V == 0 : " + str(constraints[0].dual_value))
        print("Dual v >= 0 : " + str(constraints[1].dual_value))
        print("Dual v <= vmax : " + str(constraints[2].dual_value))

    ''' (b) '''
    def part_b():
        G_star = 13.55
        G_min = 0.2*G_star

        essential_genes = []
        for i in range(n):
            v = cp.Variable((n,1))
            obj = cp.Maximize(v[-1])
            constraints = [S@v == 0, v >= 0, v <= vmax, v[i] == 0]
            prob = cp.Problem(obj, constraints)
            prob.solve()
            if v.value[-1] < G_min:
                essential_genes.append(i)
        print("essential genes: " + str(np.array(essential_genes)+1))

        synthetic_lethals = []
        for i in range(n):
            for j in range(i+1, n):
                if i in essential_genes or j in essential_genes:
                    continue
                v = cp.Variable((n,1))
                obj = cp.Maximize(v[-1])
                constraints = [S@v == 0, v >= 0, v <= vmax, v[i] == 0, v[j] == 0]
                prob = cp.Problem(obj, constraints)
                prob.solve()
                if v.value[-1] < G_min:
                    synthetic_lethals.append((i, j))
        print("synthetic lethals: " + str(np.array(synthetic_lethals)+1))

    part_a()
    part_b()

def Online_advertising_displays():
    ''' ad_disp_data.m '''
    # data for online ad display problem
    np.random.seed(0) # rand('state',0)
    n=100      #number of ads
    m=30       #number of contracts
    T=60       #number of periods

    I=10*np.random.random((T,1))  #number of impressions in each period
    R=np.random.random((n,T))    #revenue rate for each period and ad
    q=T/n*50*np.random.random((m,1))     #contract target number of impressions
    p=np.random.random((m,1))  #penalty rate for shortfall
    
    Tcontr=np.random.random((T,m))>.8 #one column per contract. 1's at the periods to be displayed
    Acontr = np.zeros((n, m))
    for i in range(n):
        contract=int(np.floor(m*np.random.rand())) # floor used instead of ceil because python arrays are 0-indexed
        Acontr[i,contract]=1 #one column per contract. 1's at the ads to be displayed
    ''''''

    N = cp.Variable((n,T))
    s = cp.Variable((m, 1))
    revenue = cp.sum(cp.multiply(R, N))
    penalty = cp.sum(cp.multiply(p, cp.maximum(0, q - cp.vstack(cp.diag(Acontr.T@N@Tcontr)))))
    profit = revenue - penalty
    obj = cp.Maximize(profit)
    constraints = [ N >= 0, cp.vstack(cp.sum(N, axis=0)) == I ]
    prob = cp.Problem(obj, constraints)
    prob.solve()
    print("Status: " + str(prob.status))
    print("Optimal net profit = " + str(profit.value))
    print("Revenue = " + str(revenue.value))
    print("Penalty = " + str(penalty.value))

    largest_revenue_ad_per_time = np.argmax(R, axis=0)
    N_largest_revenue_ad = np.zeros((n, T))
    for i in range(len(largest_revenue_ad_per_time)):
        N_largest_revenue_ad[largest_revenue_ad_per_time[i],i] = I[i]

    ''' Alternatively, only optimize the revenue
    N = cp.Variable((n,T))
    revenue = cp.sum(cp.multiply(R, N))
    obj = cp.Maximize(revenue)
    constraints = [ N >= 0, cp.vstack(cp.sum(N, axis=0)) == I ]
    prob = cp.Problem(obj, constraints)
    prob.solve()
    print("Status: " + str(prob.status))
    print("Revenue = " + str(revenue.value))
    N_largest_revenue_ad = N.value
    '''

    # revenue_largest_revenue_ad = np.sum(np.multiply(np.vstack(np.max(R, axis=0)), I)) # Another way of calculating revenue directly
    revenue_largest_revenue_ad = np.sum(np.multiply(R, N_largest_revenue_ad))
    penalty_largest_revenue_ad = np.sum(np.multiply(p, np.maximum(0, q - np.vstack(np.diag(Acontr.T@N_largest_revenue_ad@Tcontr))))) 
    profit_largest_revenue_ad = revenue_largest_revenue_ad - penalty_largest_revenue_ad
    print("Net profit if we were to display only the ad with the largest revenue per impression: ")
    print(str(profit_largest_revenue_ad))
    print("Revenue if we were to display only the ad with the largest revenue per impression: ")
    print(str(revenue_largest_revenue_ad))
    print("Penalty if we were to display only the ad with the largest revenue per impression: ")
    print(str(penalty_largest_revenue_ad))

def Ranking_by_aggregating_preferences():
    ''' rank_aggr_data.m '''
    # Data file for rank aggregation problem

    # The data was generated using the following code:
    # rand('state',0);
    # numobjects=50;
    # numpref=1000;
    # 
    # true_ranking=randperm(numobjects);      #object ids sorted according to their rank
    # map_object_to_ranking(true_ranking)=[1:numobjects]; #ranks sorted according to object id
    # objects=randint(numpref,2,numobjects)+1;  #pairs of objects in the preferences
    # 
    # #remove pairs with the same object
    # pos=find(objects(:,1)==objects(:,2));
    # while(~isempty(pos))
    #     objects(pos,:)=randint(length(pos),2,numobjects)+1;
    #     pos=find(objects(:,1)==objects(:,2));
    # end
    # 
    # #coherent preferences
    # for i=1:numpref*.9
    #     ranking1=map_object_to_ranking(objects(i,1));
    #     ranking2=map_object_to_ranking(objects(i,2));
    #     if ranking1 < ranking2
    #         preferences(i,:)=[objects(i,2) objects(i,1)];
    #     elseif ranking1 > ranking2
    #         preferences(i,:)=[objects(i,1) objects(i,2)];
    #     end
    # end
    # #incoherent measurements
    # for j=i+1:numpref
    #     ranking1=map_object_to_ranking(objects(j,1));
    #     ranking2=map_object_to_ranking(objects(j,2));
    #     if ranking1 < ranking2
    #         preferences(j,:)=[objects(j,1) objects(j,2)];
    #     elseif ranking1 > ranking2
    #         preferences(j,:)=[objects(j,2) objects(j,1)];
    #     end
    # end
    # 
    # #scramble measurements
    # preferences=preferences(randperm(numpref),:);

    n=50
    m=1000

    # one preference per row 
    # first column is i; second column is j
    preferences = np.array([
        [40,    7], [35,   22], [46,   50], [32,    8], [47,   26], [ 5,   26], [28,   10], [41,   30], [34,   21], [46,   33],
        [ 3,   44], [22,   50], [46,    8], [34,   39], [32,   48], [ 1,   39], [ 1,   28], [41,   31], [11,   21], [37,   42],
        [26,    5], [32,   30], [ 2,   27], [43,   46], [34,   47], [45,   44], [37,    9], [14,   23], [23,    2], [27,   31],
        [37,   33], [38,   31], [ 5,   10], [16,   39], [17,   50], [27,    8], [ 1,   49], [ 2,   15], [44,   48], [35,   30],
        [41,   10], [32,   23], [48,   50], [37,    4], [35,   36], [33,   16], [19,   38], [ 1,   38], [34,   47], [22,   25],
        [27,   49], [36,   19], [46,   30], [45,   22], [41,   24], [19,   21], [10,   48], [41,    9], [ 1,   41], [40,   48],
        [ 1,   48], [36,   31], [15,   21], [38,   10], [37,   49], [50,    8], [ 1,   35], [41,   35], [14,   15], [ 6,   33],
        [44,   26], [47,   49], [41,   31], [11,    2], [20,   47], [ 5,   11], [34,    7], [ 3,    4], [38,   46], [ 1,   26],
        [47,   10], [ 7,   26], [37,   34], [28,   46], [11,   16], [ 9,   28], [18,   33], [ 5,   12], [35,   49], [18,   25],
        [37,   21], [ 7,   21], [10,   19], [ 1,   32], [ 9,   26], [47,    2], [13,   30], [20,   21], [18,   12], [32,   28],
        [ 2,    5], [35,   21], [12,    7], [46,   48], [45,   31], [24,   36], [34,    3], [47,   21], [20,   18], [ 9,   29],
        [37,   24], [10,   21], [26,   40], [38,   30], [23,   15], [14,   28], [ 5,   42], [ 7,   31], [ 4,   36], [40,   24],
        [10,   31], [ 5,   43], [41,   45], [37,   26], [41,   27], [ 7,   33], [ 5,   30], [39,   31], [41,   27], [17,   31],
        [47,   29], [49,   27], [36,   26], [ 2,   42], [17,   26], [13,   49], [ 9,   15], [35,   29], [41,    3], [32,   42],
        [ 5,   27], [12,   11], [17,   34], [29,   39], [19,    8], [ 5,   50], [17,   12], [46,   30], [20,   40], [ 1,   17],
        [ 1,   45], [20,   11], [28,    9], [11,   15], [14,   41], [17,    2], [23,   16], [23,   14], [45,    4], [ 9,    8],
        [20,   38], [23,   11], [13,   28], [28,   39], [34,   14], [40,   11], [ 3,   30], [10,   49], [32,   27], [ 7,   26],
        [20,    4], [33,   10], [37,    9], [25,   49], [ 3,   27], [23,   27], [28,   10], [22,   31], [16,   50], [18,   14],
        [ 3,   30], [ 1,    3], [34,   35], [35,   21], [13,   25], [48,   21], [37,   32], [28,   24], [43,   44], [36,   42],
        [27,   10], [48,   29], [ 1,   10], [14,   26], [ 1,   11], [48,    2], [12,   28], [32,   33], [12,   44], [ 1,   33],
        [ 6,   42], [37,   38], [49,   31], [ 1,   24], [49,   29], [ 7,   24], [43,   16], [20,   42], [33,   21], [13,    9],
        [45,   11], [29,   26], [ 9,   40], [14,    8], [22,   15], [20,   22], [32,   48], [20,   45], [ 5,   39], [34,   41],
        [47,   49], [19,   39], [ 5,   29], [49,   44], [13,    2], [48,   34], [17,   38], [12,    4], [46,    2], [13,   24],
        [14,   24], [ 6,   29], [24,   12], [32,   38], [45,   22], [44,   37], [ 7,   16], [17,   44], [ 9,   38], [ 2,   30],
        [ 6,   25], [13,   25], [27,   13], [ 7,   50], [24,   13], [ 7,   10], [25,    5], [37,   10], [38,   22], [13,   23],
        [38,   33], [47,   15], [28,   16], [34,   12], [ 1,   40], [ 5,   21], [13,   50], [ 4,    5], [43,    4], [45,    1],
        [45,   32], [ 6,   20], [39,   27], [32,   35], [18,    9], [41,    8], [33,   49], [28,    7], [12,   48], [41,   29],
        [36,   30], [45,    4], [41,   32], [ 4,   48], [46,   33], [41,    8], [19,   16], [ 6,   28], [13,   10], [ 4,   40],
        [25,   16], [14,    8], [34,   42], [ 1,   19], [30,   31], [14,   50], [42,   12], [11,   33], [35,   33], [36,   39],
        [ 3,   21], [35,   50], [23,   26], [23,   44], [37,   32], [38,    2], [10,   21], [ 5,   10], [38,    2], [ 3,   50],
        [20,   31], [43,   20], [ 3,   29], [ 4,   27], [22,    8], [40,   29], [11,    8], [ 7,   31], [11,   42], [40,   36],
        [ 1,   15], [20,    7], [13,    2], [42,   36], [17,   49], [ 1,   22], [23,   12], [47,   13], [32,   43], [33,   37],
        [34,   33], [45,    6], [36,   50], [23,    4], [28,   43], [45,   31], [20,   43], [36,   39], [26,    3], [50,   42],
        [25,    1], [16,   22], [ 9,   16], [ 4,   22], [14,    4], [ 4,   29], [ 6,    3], [17,   48], [34,   28], [ 5,   45],
        [12,   16], [ 7,   29], [19,   24], [16,   27], [40,   44], [ 8,   50], [18,   32], [18,   20], [46,   22], [ 9,   47],
        [44,   27], [32,   13], [17,   19], [ 3,   19], [12,   28], [13,   28], [43,   44], [41,   10], [12,   24], [38,    7],
        [13,   41], [ 9,   26], [ 7,   29], [33,   32], [ 3,   33], [37,   12], [ 8,   28], [ 3,    7], [ 4,   48], [43,   26],
        [43,   35], [31,    5], [41,   33], [40,   11], [ 9,   12], [45,   23], [13,   29], [10,   39], [32,   24], [20,    6],
        [39,   24], [11,   21], [30,   31], [ 2,   26], [28,   16], [41,    9], [21,   45], [17,   37], [ 7,   30], [46,   25],
        [25,    1], [18,   37], [ 6,    2], [47,    8], [19,   16], [ 9,    2], [49,   39], [ 2,   47], [13,   19], [34,   23],
        [45,   25], [ 1,   28], [49,   33], [16,   29], [17,   23], [48,   29], [37,   50], [37,   29], [26,   49], [14,    8],
        [35,   36], [46,   31], [38,   15], [47,   46], [32,    4], [30,   50], [34,   15], [33,   14], [22,   15], [37,   45],
        [40,   21], [47,   38], [32,   48], [45,   14], [38,    8], [23,    3], [40,   22], [14,   44], [37,   47], [ 5,   44],
        [22,   42], [38,   20], [22,   39], [14,   19], [ 9,   16], [ 5,   33], [ 3,   37], [35,   39], [13,   25], [12,   41],
        [34,    3], [20,   14], [33,   49], [23,   19], [ 1,   39], [33,   25], [ 9,   23], [ 1,   50], [ 7,   31], [ 1,   12],
        [40,   15], [37,   19], [10,   24], [14,   16], [11,    8], [14,    4], [43,   36], [34,   44], [ 3,    7], [33,    8],
        [46,   40], [16,   27], [28,    2], [17,   31], [48,   15], [ 1,    2], [29,   39], [26,    7], [ 3,   27], [36,   21],
        [12,   35], [13,   10], [45,   36], [32,   31], [13,   31], [20,    7], [34,   47], [39,   49], [50,   39], [ 2,   25],
        [ 5,   22], [32,   16], [10,   19], [34,   49], [19,   49], [47,   19], [37,   29], [20,   15], [35,   31], [45,   49],
        [40,   29], [46,   16], [ 5,   29], [49,   25], [46,   50], [ 6,   24], [ 6,   35], [26,   30], [34,   13], [12,   39],
        [18,   24], [ 1,   32], [ 1,   36], [28,   35], [47,   44], [45,    8], [43,   35], [18,   35], [13,   35], [19,   15],
        [14,   27], [21,    5], [20,    4], [33,   26], [34,   42], [38,   46], [44,   15], [ 1,   41], [18,   11], [12,    7],
        [40,   10], [17,   13], [14,   33], [19,   16], [48,   42], [ 5,   12], [17,   11], [ 4,   25], [22,   24], [ 9,   27],
        [13,   15], [40,   50], [28,   49], [ 4,   17], [ 9,   29], [45,   21], [40,    2], [26,   39], [40,    2], [37,   49],
        [ 2,   12], [ 4,   27], [22,    8], [45,   50], [33,   19], [46,   15], [ 3,   30], [44,   15], [37,   36], [23,   32],
        [ 5,   40], [23,   15], [20,   28], [36,    8], [41,   44], [48,   24], [20,   41], [45,   35], [45,   47], [49,   12],
        [32,   39], [10,   21], [ 1,   41], [ 6,   47], [ 9,   50], [18,   12], [23,   46], [18,   49], [ 6,   16], [40,   50],
        [23,   16], [ 6,   16], [19,   48], [46,   19], [42,    8], [13,   27], [44,   31], [13,   12], [40,   48], [ 1,   13],
        [ 6,   42], [20,   32], [40,   22], [17,    4], [28,   50], [26,   42], [16,   39], [37,   39], [25,   36], [44,   24],
        [17,   49], [ 1,   44], [46,   43], [41,   18], [21,   13], [ 4,   24], [ 3,   24], [40,   16], [ 3,   15], [33,   42],
        [43,   19], [43,   42], [19,    2], [49,   42], [40,   11], [11,    3], [37,   40], [45,   18], [ 1,   36], [19,    2],
        [12,   48], [50,   39], [37,   48], [10,   21], [29,   26], [47,   38], [41,   16], [45,   42], [23,   29], [40,   16],
        [45,   50], [44,   15], [16,    7], [ 7,   21], [ 5,   39], [14,   10], [36,   39], [17,   40], [19,   27], [20,   25],
        [ 5,   48], [43,   40], [17,    5], [35,   26], [16,   42], [17,    9], [33,   39], [ 1,   46], [16,   27], [36,   17],
        [18,   16], [ 5,   23], [36,   49], [ 6,    7], [45,   48], [48,   18], [23,   26], [14,   25], [42,   23], [44,   42],
        [37,   41], [37,   24], [25,   24], [11,    3], [43,   50], [19,   30], [12,    8], [44,   31], [27,   15], [33,   15],
        [ 3,   21], [32,   44], [ 1,   25], [34,   10], [ 3,    2], [47,   44], [37,   44], [18,   42], [16,   50], [13,   36],
        [20,    8], [44,   48], [19,    2], [37,   41], [18,    8], [28,   44], [13,   32], [35,   25], [ 6,   32], [11,   49],
        [ 9,   39], [ 9,   33], [22,   49], [36,   11], [26,   25], [18,   31], [47,   42], [ 5,   45], [ 9,   47], [48,   50],
        [34,   48], [20,   19], [35,   26], [ 1,   42], [ 5,   42], [ 1,   35], [45,   36], [42,   48], [35,   19], [25,   43],
        [ 5,   40], [18,   21], [25,    1], [41,   21], [28,   48], [37,   43], [ 9,   32], [45,   21], [48,    2], [25,   21],
        [ 1,   49], [19,   26], [ 7,   27], [27,    3], [ 1,   32], [10,   15], [ 7,   43], [21,   42], [ 7,   30], [ 5,   49],
        [ 6,   10], [49,   30], [17,   45], [18,   37], [41,   42], [19,   49], [11,   10], [16,   21], [34,    4], [49,   42],
        [11,   29], [30,   23], [ 9,   14], [17,   35], [19,   31], [16,   22], [38,   21], [17,   33], [46,   33], [17,    4],
        [30,   24], [ 3,   23], [12,   11], [13,   20], [35,   33], [34,   50], [32,    8], [12,   50], [10,   31], [35,   30],
        [37,   30], [20,   26], [11,   24], [14,   10], [ 7,   17], [49,   31], [ 1,   21], [12,   19], [19,   29], [23,   36],
        [15,   26], [10,   31], [21,   38], [12,    7], [35,   21], [ 3,   48], [ 3,   30], [12,   38], [ 2,   39], [17,   22],
        [37,   33], [ 7,   36], [21,   34], [17,   24], [48,   25], [20,   50], [35,   42], [43,   15], [ 3,   22], [ 5,   42],
        [18,   46], [16,   40], [13,   20], [22,   24], [33,   29], [ 5,   26], [34,   36], [35,   29], [ 1,   10], [23,   11],
        [32,   42], [ 1,    8], [22,   31], [12,   33], [16,   41], [19,   31], [28,   16], [14,   30], [ 1,   24], [11,   48],
        [20,   31], [50,   15], [15,    8], [18,   11], [12,   15], [13,   33], [43,   30], [32,   36], [ 5,   41], [20,   24],
        [ 4,   49], [28,   50], [43,   36], [37,    6], [17,   41], [37,   39], [13,    7], [35,   21], [34,   11], [23,    4],
        [ 2,    9], [41,   23], [48,   25], [36,   24], [47,   46], [44,   25], [49,   32], [ 9,   23], [ 3,   31], [29,   27],
        [33,   48], [32,    1], [39,   27], [39,   50], [ 7,   15], [ 7,   49], [35,   21], [21,   42], [ 5,   41], [10,   22],
        [ 5,    3], [32,   43], [41,   30], [45,   34], [16,   44], [ 9,   42], [ 7,   25], [34,   23], [ 9,    3], [18,   10],
        [37,   45], [44,   30], [33,   34], [34,   29], [34,    5], [ 4,   29], [46,   15], [41,    8], [11,    3], [34,   36],
        [47,   48], [34,   26], [23,   21], [ 7,   31], [ 1,   27], [28,   29], [37,   19], [ 6,   35], [39,   34], [47,   25],
        [17,   29], [19,   22], [48,   22], [46,   24], [19,   25], [ 7,   36], [19,   31], [34,   32], [17,   47], [20,   32],
        [10,   22], [47,   19], [ 3,   35], [39,   22], [49,   41], [ 1,   32], [14,   44], [ 1,   42], [41,   44], [14,   40],
        [ 9,   15], [ 6,   11], [41,   42], [42,   34], [ 9,   44], [12,   19], [ 6,   42], [43,   24], [32,   21], [16,   39],
        [23,   29], [23,   16], [ 9,   25], [46,   27], [18,   20], [18,   17], [40,   46], [14,   24], [ 1,    9], [ 1,   25],
        [49,   15], [48,   21], [28,   38], [35,   39], [32,   33], [ 9,   10], [ 3,   35], [16,   50], [ 4,   24], [44,   17],
        [15,   31], [34,   21], [33,   27], [48,   32], [47,   50], [23,   15], [47,   26], [37,   39], [ 9,   19], [ 5,   25],
        [13,   25], [32,   22], [ 4,   10], [ 3,   44], [31,   18], [28,   19], [11,   28], [19,   25], [35,   30], [23,   48],
        [ 9,   16], [25,    8], [14,   22], [ 1,   41], [34,    3], [44,   30], [23,    9], [32,    2], [ 1,   11], [28,   37],
        [14,   38], [38,   42], [32,   42], [33,   25], [ 8,   24], [ 3,    4], [ 4,   39], [29,   25], [18,   20], [41,   47],
        [34,   15], [ 9,    4], [ 6,   22], [23,   48], [14,   48], [41,   49], [ 6,   32], [30,   50], [16,   45], [12,   36],
        [44,   50], [12,   16], [ 7,   48], [42,    8], [30,   42], [47,   43], [19,   30], [ 6,   26], [15,   24], [30,   15],
        [37,    4], [ 3,   38], [14,   26], [41,   11], [ 4,   42], [12,   32], [44,   20], [ 8,   24], [45,   29], [46,   25],
        [13,   50], [ 7,   19], [49,   30], [47,    4], [45,   19], [10,   48], [ 7,   48], [16,   27], [40,   21], [14,   47]])
    ''''''

    '''
    The following is not a convex constraint
    constraints = [ v[i] == cp.max(r[preferences[i,1]-1] + 1 - r[preferences[i,0]-1], 0) for i in range(m)] # matlab arrays are 1-indexed
    We convert it into matrix form
    '''
    preference_matrix = np.zeros((m,n))
    for i in range(len(preferences)):
        preference_matrix[i, preferences[i, 0]-1] = -1
        preference_matrix[i, preferences[i, 1]-1] = 1

    large_violation = 2.0

    r = cp.Variable((n,1))
    v = cp.maximum(preference_matrix@r + 1, 0)
    obj_a = cp.Minimize(cp.sum_squares(v))
    prob_a = cp.Problem(obj_a)
    prob_a.solve()
    a_v = v.value
    a_violations = np.sum(a_v>0.001)
    print("(a) # preference violations = " + str(a_violations))
    print("(a) violation greater than " + str(large_violation) + " = " + str(np.sum(a_v > large_violation)))
    # print("(a) total preference violation = " + str(np.sum(a_v)))

    obj_b = cp.Minimize(cp.sum(v))
    prob_b = cp.Problem(obj_b)
    prob_b.solve()
    b_v = v.value
    b_violations = np.sum(b_v>0.001)
    print("(b) preference violations = " + str(b_violations))
    print("(b) violation greater than " + str(large_violation) + " = " + str(np.sum(b_v > large_violation)))
    # print("(b) total preference violation = " + str(np.sum(b_v)))

    print("(a) has " + str(a_violations - b_violations) + " more violations than (b)")
    
# Gradient_and_Newton_methods()
# Flux_balance_analysis_in_systems_biology()
# Online_advertising_displays()
Ranking_by_aggregating_preferences()
