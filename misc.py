import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import chain, combinations

def pure_filter(candidate_sets):
    return candidate_sets
    ret_set = []

    for candidate_set in candidate_sets:
        duplicate_elem = []
        for elem in candidate_set:
            for compare_set in candidate_sets:
                if set(compare_set) == set(candidate_set):
                    continue
                if elem in compare_set:
                    duplicate_elem.append(elem)
                    break

        if set(duplicate_elem) != set(candidate_set):
            ret_set.append(candidate_set)

    return ret_set
        

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

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
            if set(candidate_set) <= set(max_cliq) and not (set(superset) <= set(max_cliq)):
                exist_flag = False
                break
        if exist_flag:
            return True
    return False

def is_edge_cover(smaller_sets, whole_sets):
    nb_all_elements = len(set().union(*whole_sets))

    whole_adj = np.zeros((nb_all_elements, nb_all_elements))
    smaller_adj = np.zeros((nb_all_elements, nb_all_elements))

    for smaller_set in smaller_sets:
        for i in smaller_set:
            for j in smaller_set:
                smaller_adj[i, j] = 1
                smaller_adj[j, i] = 1

    for whole_set in whole_sets:
        for i in whole_set:
            for j in whole_set:
                whole_adj[i, j] = 1
                whole_adj[j, i] = 1

    if np.linalg.norm(smaller_adj - whole_adj) == 0:
        return True
    else:
        return False

def is_complete_collection(candidate_collention, list_maximal_cliques):
    for maximal_cliques_intervene in list_maximal_cliques:
        shattered_set = []
        for maximal_clique in maximal_cliques_intervene:
            maximal_sets_with_intersections = []
            for maximal_valid_set in candidate_collention:
                if not set(maximal_valid_set).isdisjoint(maximal_clique):
                    maximal_sets_with_intersections.append(maximal_valid_set)

            for collection_with_intersection in powerset(maximal_sets_with_intersections):
                if set().union(*collection_with_intersection) == set(maximal_clique):
                    shattered_set.append(maximal_clique)
            
        if not is_edge_cover(shattered_set, maximal_cliques_intervene):
            return False
        
    return True