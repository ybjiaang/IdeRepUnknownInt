import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

np.random.seed(2)

def show_graph_with_labels(adjacency_matrix):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=500)
    plt.show()

def create_graph(adjacency_matrix):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)

    return gr


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

        print(self.adj)

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

            # generate_weights (might be unstable to generate weights using gaussian. because some weights might be too small)
            # self.weights = np.abs(np.random.normal(size=self.weights.shape))
            self.weights = np.abs(np.random.uniform(low=-1, high=1, size=self.weights.shape))
            self.weights *= self.adj
            self.weights /= np.linalg.norm(self.weights, axis=1, keepdims=True)

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
        print(self.adj)
    
    def get_random_latent_dag(self):
        # nothing to be done here
        if self.graphtype == 'purechild':
            self.perm = np.arange(self.num_hidden)
            np.random.shuffle(self.perm) # topsort
            for i in range(self.num_hidden):
                for j in range(i + 1, self.num_hidden):
                    if np.random.binomial(1, p = self.edge_prob) == 1:
                        self.adj[self.perm[j], self.perm[i]] = 1

            # generate_weights
            # only positive weights
            # self.weights = np.abs(np.random.normal(size=self.weights.shape))
            self.weights = np.abs(np.random.uniform(low=-1, high=1, size=self.weights.shape))
            self.weights *= self.adj
            for i in range(self.num_hidden):
                norm = np.linalg.norm(self.weights[i,:])
                if norm > 0:
                    self.weights[i,:] /= norm
            
    
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
    def __init__(self, num_hidden, num_observed, epsilon = 0.1, latent_dag_density = 0.5, bip_graph_density = 0.5):
        self.num_hidden = num_hidden
        self.num_observed = num_observed

        self.latentdag = LatentDAG(num_hidden, epsilon = epsilon, edge_prob = latent_dag_density)
        self.bipgraph = BipGraph(num_hidden, num_observed, epsilon = epsilon, prob = bip_graph_density)

    def generate_samples(self, n, int_target):
        # If int_target is -1, then obversational distributions, otherwise, it is one of the hidden variables

        assert(-1 <= int_target < self.num_hidden)

        latents = self.latentdag.generate_samples(n, int_target)
        observed = self.bipgraph.generate_samples(latents)

        return observed


num_hidden = 3
num_observed = 4
num_samples = 1000
MMgraph = Latent_and_Bipartite_graph(num_hidden, num_observed)

list_covs = []
list_maximal_cliques = []
for i in range(-1, num_observed-1):
    observed = MMgraph.generate_samples(n=num_samples, int_target=i)
    cov = np.corrcoef(observed)
    list_covs.append(cov)

    # create the adj for udg
    udg_adj = (np.abs(cov) >= 0.1).astype(int)

    # used for debugging
    # show_graph_with_labels(adj)

    # find all the maximal cliques
    gr = create_graph(udg_adj)
    list_maximal_cliques.append(list(nx.find_cliques(gr)))



