import numpy as np
import cvxpy as cp
import scipy.sparse as sp
import scipy.stats as st
import pdb

def Maximum_likelihood_prediction_of_team_ability():
    ''' team_data.m '''
    n = 10

    m = 45

    m_test = 45

    sigma= 0.250

    train=np.array([
        [1,2,1],
        [1,3,1],
        [1,4,1],
        [1,5,1],
        [1,6,1],
        [1,7,1],
        [1,8,1],
        [1,9,1],
        [1,10,1],
        [2,3,-1],
        [2,4,-1],
        [2,5,-1],
        [2,6,-1],
        [2,7,-1],
        [2,8,-1],
        [2,9,-1],
        [2,10,-1],
        [3,4,1],
        [3,5,-1],
        [3,6,-1],
        [3,7,1],
        [3,8,1],
        [3,9,1],
        [3,10,1],
        [4,5,-1],
        [4,6,-1],
        [4,7,1],
        [4,8,1],
        [4,9,-1],
        [4,10,-1],
        [5,6,1],
        [5,7,1],
        [5,8,1],
        [5,9,-1],
        [5,10,1],
        [6,7,1],
        [6,8,1],
        [6,9,-1],
        [6,10,-1],
        [7,8,1],
        [7,9,1],
        [7,10,-1],
        [8,9,-1],
        [8,10,-1],
        [9,10,1]])

    test=np.array([
        [1,2,1],
        [1,3,1],
        [1,4,1],
        [1,5,1],
        [1,6,1],
        [1,7,1],
        [1,8,1],
        [1,9,1],
        [1,10,1],
        [2,3,-1],
        [2,4,1],
        [2,5,-1],
        [2,6,-1],
        [2,7,-1],
        [2,8,1],
        [2,9,-1],
        [2,10,-1],
        [3,4,1],
        [3,5,-1],
        [3,6,1],
        [3,7,1],
        [3,8,1],
        [3,9,-1],
        [3,10,1],
        [4,5,-1],
        [4,6,-1],
        [4,7,-1],
        [4,8,1],
        [4,9,-1],
        [4,10,-1],
        [5,6,-1],
        [5,7,1],
        [5,8,1],
        [5,9,1],
        [5,10,1],
        [6,7,1],
        [6,8,1],
        [6,9,1],
        [6,10,1],
        [7,8,1],
        [7,9,-1],
        [7,10,1],
        [8,9,-1],
        [8,10,-1],
        [9,10,1]])
    ''''''
    # A = sparse(1:m,train(:,1),train(:,3),m,n) + ...
        # sparse(1:m,train(:,2),-train(:,3),m,n);
    A = (sp.csr_matrix((train[:,2], (range(m), train[:,0]-1)), shape=(m,n)) + # -1 because matlab is 1-indexed
            sp.csr_matrix((-train[:,2], (range(m), train[:,1]-1)), shape=(m,n))) # -1 because matlab is 1-indexed
    a = cp.Variable((n,1))

    ''' cvxpy is missing log_normcdf
    obj = cp.Maximize(cp.sum([cp.log_normcdf(A[i]@a, 0, sigma) for i in range(m)]))
    constraints = [ a <= 1, a >= 0 ]
    prob = cp.Problem(obj, constraints)
    prob.solve()
    print("Status: " + str(prob.status))
    '''


def Allocation_of_interdiction_effort():
    ''' interdict_alloc_data.m '''
    # rand('state',0);
    n=10
    m=20
    edges=np.hstack([np.array([[1,1,1,2,2,2,3,3,4,4,5,5,6,6,7,7,8,7,8,9]]).T,
        np.array([[2,3,4,6,3,4,5,6,6,7,8,7,7,8,8,9,9,10,10,10]]).T])
    A=np.zeros((n,m))
    for j in range(edges.shape[0]):
        A[edges[j,0]-1,j]=-1
        A[edges[j,1]-1,j]=1
    a=2*np.random.rand(m,1);
    x_max = 1+np.random.rand(m,1)
    B=m/2;
    ''''''

    def find_prev_node_and_edge(current_node):
        return [(edge, node_pair[0]-1) for (edge, node_pair) in enumerate(edges.tolist()) if node_pair[1]-1 == current_node]

    def optimal_allocation():
        P = cp.Variable((n,1), name="P")
        x = cp.Variable((m,1), name="x")
        log_p = -cp.multiply(a, x)
        edge_budget_constraint = [x >= 0, x <= x_max]
        total_budget_constraint = [cp.sum(x) <= B]
        source_node_logprob_constraint = [P[0] == np.log(1)]

        node_logprob_constraints = [ P[this_node] >= cp.max(cp.vstack([P[prev_node] + log_p[edge_from_prev] for (edge_from_prev, prev_node) in find_prev_node_and_edge(this_node)])) for this_node in range(1,n)]

        obj = cp.Minimize(P[-1])
        constraints = edge_budget_constraint + total_budget_constraint + source_node_logprob_constraint + node_logprob_constraints
        prob = cp.Problem(obj, constraints)
        prob.solve()
        print("Status: " + str(prob.status))
        print("optimal P_max = " + str(np.exp(P.value[-1])))

    def uniform_allocation():
        P = cp.Variable((n,1), name="P")
        x = np.ones((m,1))*B/m
        log_p = -np.multiply(a, x)
        source_node_logprob_constraint = [P[0] == np.log(1)]

        node_logprob_constraints = [ P[this_node] >= cp.max(cp.vstack([P[prev_node] + log_p[edge_from_prev] for (edge_from_prev, prev_node) in find_prev_node_and_edge(this_node)])) for this_node in range(1,n)]

        obj = cp.Minimize(P[-1])
        constraints = source_node_logprob_constraint + node_logprob_constraints
        prob = cp.Problem(obj, constraints)
        prob.solve()
        print("Status: " + str(prob.status))
        print("uniform P_max = " + str(np.exp(P.value[-1])))

    optimal_allocation()
    uniform_allocation()
    # code to plot the graph (if you have biograph)
    #G=sparse(edges(:,1),edges(:,2),1,n,n);
    #view(biograph(G));

# Maximum_likelihood_prediction_of_team_ability()
Allocation_of_interdiction_effort()
