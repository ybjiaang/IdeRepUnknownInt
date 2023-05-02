import numpy as np
import networkx as nx

"""
Notation for adjancy matrix
If X_i -> X_j, then A[j][i] = 1. In other words, the ith row of the matrix is all the parents of X_i
"""



class BipGraph():
    """
    The class that contains infomation about the bipartite causal graph between 
    observed and hidden variables
    """
    def __init__(self, num_hidden, num_observed, prob = 0.5, epsilon=1, graphtype='purechild'):
        self.num_hidden = num_hidden
        self.num_observed = num_observed
        self.adj = np.zeros((self.num_observed, self.num_hidden))
        self.weights = np.zeros_like(self.adj)
        self.graphtype = graphtype
        self.epsilon = epsilon
        self.prob = prob

        self.gen_random_graph()

        # print(self.adj)

    def gen_random_graph(self, attempts = 1000):
        if self.graphtype == 'purechild':
            if self.num_hidden >= self.num_observed:
                raise ValueError('num of hidden is not smaller than the num of observed')

            # generate a random biparite graph first
            self.adj = np.random.multinomial(1, [1/self.num_hidden]*self.num_hidden, size=self.num_observed)
            self.adj = self.adj | np.random.binomial(1, self.prob, size = self.adj.shape)
            for i in range(self.num_hidden):
                for j in range(self.num_hidden):
                    self.adj[i][j] = (i == j)

            # randomize the columns
            np.random.shuffle(np.transpose(self.adj))
        
        if self.graphtype == 'singlesource':
            if self.num_hidden >= self.num_observed:
                raise ValueError('num of hidden is not smaller than the num of observed')
            
            self.adj = np.random.multinomial(1, [1/self.num_hidden]*self.num_hidden, size=self.num_observed)
            self.adj = self.adj | np.random.binomial(1, self.prob, size = self.adj.shape)

        # generate_weights (might be unstable to generate weights using gaussian. because some weights might be too small)
        # 0.5 - 2
        self.weights = np.abs(np.random.uniform(low=0.5, high=2, size=self.weights.shape)) #* np.random.choice([-1, 1], size=self.weights.shape, p=[1./2, 1./2])
        self.weights *= self.adj

    def generate_samples(self, latents):
        _, n = latents.shape

        observed = self.weights @ latents + np.random.normal(0,self.epsilon, (self.num_observed, n))
        
        return observed

class LatentDAG():
    def __init__(self, num_hidden, edge_prob = 0.5, epsilon=1, graphtype='purechild'):
        self.num_hidden = num_hidden
        self.adj = np.zeros((self.num_hidden, self.num_hidden))
        self.weights = np.zeros_like(self.adj)
        self.edge_prob = edge_prob
        self.graphtype = graphtype
        self.epsilon = epsilon

        self.get_random_latent_dag()
        # print(self.adj)
    
    def get_random_latent_dag(self):
        # nothing to be done here
        if self.graphtype == 'purechild':
            self.perm = np.arange(self.num_hidden)
            np.random.shuffle(self.perm) # topsort
            for i in range(self.num_hidden):
                for j in range(i + 1, self.num_hidden):
                    if np.random.binomial(1, p = self.edge_prob) == 1:
                        self.adj[self.perm[j], self.perm[i]] = 1

        if self.graphtype == 'singlesource':
            self.perm = np.arange(self.num_hidden)
            np.random.shuffle(self.perm) # topsort
            for i in range(1, self.num_hidden):
                for j in range(i + 1, self.num_hidden):
                    if np.random.binomial(1, p = self.edge_prob) == 1:
                        self.adj[self.perm[j], self.perm[i]] = 1
                if sum(self.adj[self.perm[j]]) == 0:
                    self.adj[self.perm[0], self.perm[j]] = 1
            


        # generate_weights
        # only positive weights
        self.weights = np.random.uniform(low=0.5, high=2, size=self.weights.shape) #* np.random.choice([-1, 1], size=self.weights.shape, p=[1./2, 1./2])
        self.weights *= self.adj
        # for i in range(self.num_hidden):
        #     norm = np.linalg.norm(self.weights[i,:])
        #     if norm > 0:
        #         self.weights[i,:] /= norm
            
    
    def generate_samples(self, n, int_target):
        latents = np.zeros((self.num_hidden, n))
        for i in self.perm:
            if i == int_target:
                latent = np.random.normal(0,self.epsilon,n)
            else:
                latent = self.weights[i, :] @ latents + np.random.normal(0,self.epsilon,n)
            latents[i,:] = latent
        
        return latents


class Latent_and_Bipartite_graph():
    """
    Class that combines information about Latent and Bipartite causal graphs
    """
    def __init__(self, num_hidden, num_observed, epsilon = 0.1, latent_dag_density = 0.5, bip_graph_density = 0.6, mode='purechild'):
        self.num_hidden = num_hidden
        self.num_observed = num_observed

        self.latentdag = LatentDAG(num_hidden, epsilon = epsilon, edge_prob = latent_dag_density, graphtype = mode)
        self.bipgraph = BipGraph(num_hidden, num_observed, epsilon = epsilon, prob = bip_graph_density, graphtype = mode)

    def generate_samples(self, n, int_target, puregraph = False):
        # If int_target is -1, then obversational distributions, otherwise, it is one of the hidden variables

        assert(-1 <= int_target < self.num_hidden)

        if puregraph:

            intervene_adj = np.copy(self.latentdag.adj)
            if int_target >= 0:
                intervene_adj[int_target, :] = 0

            B_true = intervene_adj + np.identity(self.num_hidden).astype(int) 
            B = B_true

            for _ in range(self.num_hidden):
                B = B_true @ B
            
            B = np.identity(self.num_hidden) @ B @ np.identity(self.num_hidden)
            connecting_B = np.identity(self.num_hidden)
            for i in range(self.num_hidden):
                for j in range(i+1, self.num_hidden):
                    row_i = B_true[i,:]
                    row_j = B_true[j,:]

                    connecting_B[i, j] = row_i @ B @ row_j + row_j @ B @ row_i
                    connecting_B[j, i] = connecting_B[i, j]

            udj_cov = np.identity(self.num_observed)
            for i in range(self.num_observed):
                for j in range(i+1, self.num_observed):
                    row_i = self.bipgraph.adj[i,:]
                    row_j = self.bipgraph.adj[j,:]

                    udj_cov[i, j] = row_i @ connecting_B @ row_j + row_j @ connecting_B @ row_i
                    udj_cov[j, i] = udj_cov[i, j]

            return (udj_cov > 0).astype(int)

        else:
            latents = self.latentdag.generate_samples(n, int_target)
            observed = self.bipgraph.generate_samples(latents)

            return observed