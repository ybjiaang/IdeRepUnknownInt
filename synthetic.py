import numpy as np
import networkx as nx
from misc import powerset, show_graph_with_labels, create_graph, is_valid, exist_superset, not_maximal, is_complete_collection

from graph import Latent_and_Bipartite_graph

np.random.seed(3)

def find_maximal_cliques(MMgraph):
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

    return list_maximal_subsets, set_all_maximal_cliques, list_maximal_cliques

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


num_hidden = 3
num_observed = 4
num_samples = 1000


nb_experiments = 5

for _ in range(nb_experiments):
    # Create graph
    MMgraph = Latent_and_Bipartite_graph(num_hidden, num_observed)

    # Find maximal cliques and maximal subsets
    list_maximal_subsets, set_all_maximal_cliques, list_maximal_cliques = find_maximal_cliques(MMgraph)

    # Identify the bipartite graph
    biparG = identify_bipartie_graph(list_maximal_subsets, set_all_maximal_cliques)
    print("biparite graph")
    print(biparG)

    # Identify the latent graph
    latentG = identify_latent_graph(biparG, list_maximal_cliques)
    print("latent graph")
    print(latentG)

    # Evaluation