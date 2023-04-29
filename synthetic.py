import numpy as np
import networkx as nx
from misc import powerset, show_graph_with_labels, create_graph

np.random.seed(3)

def is_valid(candidate_clique, list_maximal_cliques):
    for maximal_cliques in list_maximal_cliques:
        exists_flag = False
        for maximal_clique in maximal_cliques:
            if set(candidate_clique) <= set(maximal_clique):
                exists_flag = True
        if not exists_flag:
            return False
    return True

def exist_superset(canidate_set, list_sets, proper = True):
    supersets = []
    for one_set in list_sets:
        if proper:
            if set(canidate_set) < set(one_set):
                supersets.append(list(one_set))
        else:
            if set(canidate_set) <= set(one_set):
                supersets.append(list(one_set))
    return supersets

def not_maximal(candidate_set, list_valid_subsets, set_all_maximal_cliques):
    # check if a subset of another set
    supersets = exist_superset(candidate_set, list_valid_subsets)
    if len(supersets) == 0:
        return False

    for superset in supersets:
        exist_flag = True
        for max_cliq in set_all_maximal_cliques:
            if set(candidate_set) > set(max_cliq) or set(superset) > set(max_cliq):
                exist_flag = False
                break
        if exist_flag:
            return True
    return False

def is_complete_collection(candidate_collention, set_all_maximal_cliques):
    for maximal_clique in set_all_maximal_cliques:
        maximal_sets_with_intersections = []
        for maximal_valid_set in candidate_collention:
            if not set(maximal_valid_set).isdisjoint(maximal_clique):
                maximal_sets_with_intersections.append(maximal_valid_set)
        
        exist_flag = False
        for collection_with_intersection in powerset(maximal_sets_with_intersections):
            if set().union(*collection_with_intersection) == set(maximal_clique):
                exist_flag = True
                break
        if not exist_flag:
            return False
    return True

def identify_bipartie_graph(list_maximal_subsets, set_all_maximal_cliques, mode="purechild"):
    if mode == 'purechild':
        for candidate_collention in powerset(list_maximal_subsets):      
            if is_complete_collection(candidate_collention, set_all_maximal_cliques):
                return candidate_collention
        
        raise ValueError("Bipartie graph cannot be identified assuming pure child")

def identify_latent_graph(biparG, list_maximal_cliques):
    nb_estimated_latents = len(biparG)
    estimated_adj_matrix = np.zeros((nb_estimated_latents, nb_estimated_latents))
    intervention_target_set = set()

    # step 0: create mG list
    mG = []
    for maximal_clique in list_maximal_cliques:
        mG_per_inter = []
        for i in range(nb_estimated_latents):
            for j in range(i+1, nb_estimated_latents):
                union = set(biparG[i] + biparG[j])
                if len(exist_superset(union, maximal_clique, proper=False)) == 0:
                    mG_per_inter.append((i, j))

        mG.append(mG_per_inter)

    # step 1: remove any pairs that has appeared twice

    count_matrix = np.zeros((nb_estimated_latents, nb_estimated_latents))
    for mG_per_inter in mG:
        for pair in mG_per_inter:
            count_matrix[pair[0]][pair[1]] += 1
            count_matrix[pair[1]][pair[0]] += 1

    to_delete = []
    for i in range(nb_estimated_latents): 
        for j in range(i+1, nb_estimated_latents):
            if count_matrix[i][j] > 1:
                to_delete.append((i, j))

    mG_deleted = []
    for mG_per_inter in mG:
        mG_deleted.append(list(x for x in mG_per_inter if x not in to_delete))

    # step 2: Add edges for colliders
    for mG_per_inter in mG_deleted:
        if len(mG_per_inter) > 1:
            try:
                common_element = list(set(mG_per_inter[0]).intersection(set(mG_per_inter[1])))[0]
            except:
                raise ValueError("no common element")

            intervention_target_set.add(common_element)
            for v_tuple in mG_per_inter:
                h_1 = v_tuple[0]
                h_2 = v_tuple[1]

                if h_1 == common_element:
                    estimated_adj_matrix[h_1][h_2] = 1
                else:
                    estimated_adj_matrix[h_2][h_1] = 1
            
            mG_deleted.remove(mG_per_inter)

    # step 3: Add compelled edges
    newInv = True
    while newInv:
        newInv = False

        for mG_per_inter in mG_deleted:
            assert(len(mG_per_inter) <= 1)

            if len(mG_per_inter) == 1:
                v_tuple = mG_per_inter[0]
                h_1 = v_tuple[0]
                h_2 = v_tuple[1]        

                if h_1 in intervention_target_set:
                    estimated_adj_matrix[h_2][h_1] = 1
                    intervention_target_set.add(h_2)
                    newInv = True
                    mG_deleted.remove(mG_per_inter)
                    continue
                if h_2 in intervention_target_set:
                    estimated_adj_matrix[h_1][h_2] = 1
                    intervention_target_set.add(h_1)
                    newInv = True
                    mG_deleted.remove(mG_per_inter)
                    continue

    # step 3: Add unoriented edges
    for mG_per_inter in mG_deleted:
        assert(len(mG_per_inter) <= 1)

        if len(mG_per_inter) == 1:
            v_tuple = mG_per_inter[0]
            h_1 = v_tuple[0]
            h_2 = v_tuple[1]

            estimated_adj_matrix[h_2][h_1] = 1
            estimated_adj_matrix[h_1][h_2] = 1

    return estimated_adj_matrix

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
            # 0.5 - 2
            self.weights = np.abs(np.random.uniform(low=0.25, high=1, size=self.weights.shape))
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
            self.weights = np.abs(np.random.uniform(low=0.25, high=1, size=self.weights.shape))
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
    def __init__(self, num_hidden, num_observed, epsilon = 0.1, latent_dag_density = 0.9, bip_graph_density = 0.5):
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

# create udgs and find all maximal cliques
list_covs = []
list_maximal_cliques = []
set_all_maximal_cliques = [] # unique set of maxmial cliques
for i in range(-1, num_observed-1):
    observed = MMgraph.generate_samples(n=num_samples, int_target=i)
    cov = np.corrcoef(observed)
    list_covs.append(cov)

    # create the adj for udg
    udg_adj = (np.abs(cov) >= 0.1).astype(int)

    # used for debugging
    # show_graph_with_labels(udg_adj)

    # find all the maximal cliques
    gr = create_graph(udg_adj)
    max_clique_list = list(nx.find_cliques(gr))

    for max_clique in max_clique_list:
        if max_clique not in set_all_maximal_cliques:
            set_all_maximal_cliques.append(max_clique)
    if max_clique_list not in list_maximal_cliques:
        list_maximal_cliques.append(max_clique_list)

# print(list_maximal_cliques)
# print(set_all_maximal_cliques)


# find all maxmial valid subsets
list_valid_subsets = []
for candidate_set in powerset(range(num_observed)):
    if len(candidate_set) == 0:
        continue
    else:
        if is_valid(candidate_set, list_maximal_cliques):
            list_valid_subsets.append(list(candidate_set))

list_maximal_subsets = []
for candidate_set in list_valid_subsets:
    if not not_maximal(candidate_set, list_valid_subsets, set_all_maximal_cliques):
        list_maximal_subsets.append(candidate_set)

"""
identify bipartite graph
"""
biparG = identify_bipartie_graph(list_maximal_subsets, set_all_maximal_cliques)
print("biparite graph")
print(biparG)

"""
identify latent graph
"""
latentG = identify_latent_graph(biparG, list_maximal_cliques)
print("latent graph")
print(latentG)

"""
evaluation
"""