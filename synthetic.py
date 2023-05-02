import numpy as np
import networkx as nx
from misc import powerset, show_graph_with_labels, create_graph, is_valid, exist_superset, not_maximal, is_complete_collection, pure_filter
import itertools
from metric import get_metrics
from tqdm import tqdm
from collections import Counter
import argparse

from graph import Latent_and_Bipartite_graph

np.random.seed(3)

def find_maximal_cliques(MMgraph, options):
    # create udgs and find all maximal cliques
    list_covs = []
    list_maximal_cliques = []
    set_all_maximal_cliques = [] # unique set of maxmial cliques
    num_hidden = MMgraph.num_hidden
    num_observed = MMgraph.num_observed
    for i in range(-1, num_hidden):
        if options.puregraph:
            udg_adj = MMgraph.generate_samples(n=num_samples, int_target=i, puregraph=options.puregraph)
            print(udg_adj)
        else:
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
    print("-----")
    print(list_maximal_cliques)
    return list_maximal_subsets, set_all_maximal_cliques, list_maximal_cliques

def identify_bipartie_graph(list_maximal_subsets, set_all_maximal_cliques, mode="purechild"):
    list_maximal_subsets_non_redudant = []
    for maximal_subset in list_maximal_subsets:
        if len(exist_superset(maximal_subset, list_maximal_subsets)) == 0:
            list_maximal_subsets_non_redudant.append(maximal_subset)

    if mode == 'purechild':
        # print("maximal subsets")
        # print(list_maximal_subsets)
        # print(set_all_maximal_cliques)
        for candidate_collention in powerset(list_maximal_subsets):  
            if is_complete_collection(candidate_collention, set_all_maximal_cliques):
                return pure_filter(candidate_collention)
        
        # raise ValueError("Bipartie graph cannot be identified assuming pure child")
        return pure_filter(list_maximal_subsets_non_redudant)
    
    if mode == 'singlesource':
        return list_maximal_subsets_non_redudant

def identify_latent_graph(biparG, list_maximal_cliques):
    # print("identifying latent graph")
    nb_estimated_latents = len(biparG)
    estimated_adj_matrix = np.zeros((nb_estimated_latents, nb_estimated_latents))
    intervention_target_set = set()

    # step 0: create mG list
    mG = []

    # print(list_maximal_cliques)
    for maximal_clique in list_maximal_cliques:
        mG_per_inter = []
        for i in range(nb_estimated_latents):
            for j in range(i+1, nb_estimated_latents):
                union = set(biparG[i] + biparG[j])
                # print(i, j)
                # print(union)
                # print(maximal_clique)
                if len(exist_superset(union, maximal_clique, proper=False)) == 0:
                    mG_per_inter.append((i, j))
                    # if there is only one intervene distribution, put everything twice
                    if len(list_maximal_cliques) == 1:
                        mG_per_inter.append((i, j))

        mG.append(mG_per_inter)
    # print(mG)
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

    # print("after step 1")
    # print(estimated_adj_matrix)
    # step 2: Add edges for colliders
    set_mG_per_inter = []
    for mG_per_inter in mG_deleted:
        if len(mG_per_inter) > 1:
            try:
                all_elements = []
                for mg_tuple in mG_per_inter:
                    all_elements += mg_tuple

                common_element = Counter(all_elements).most_common()[0][0]
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
            
            set_mG_per_inter.append(mG_per_inter)

    for mG_per_inter in set_mG_per_inter:
        mG_deleted.remove(mG_per_inter)

    # print("after step 2")
    # print(estimated_adj_matrix)
    # step 3: Add compelled edges
    newInv = True
    while newInv:
        newInv = False


        set_mG_per_inter = []
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
                    set_mG_per_inter.append(mG_per_inter)
                    continue
                if h_2 in intervention_target_set:
                    estimated_adj_matrix[h_1][h_2] = 1
                    intervention_target_set.add(h_1)
                    newInv = True
                    set_mG_per_inter.append(mG_per_inter)
                    continue

        for mG_per_inter in set_mG_per_inter:
            mG_deleted.remove(mG_per_inter)

    # print("after step 3")
    # print(estimated_adj_matrix)
    # print(mG_deleted)
    # step 4: Add unoriented edges
    for mG_per_inter in mG_deleted:
        assert(len(mG_per_inter) <= 1)

        if len(mG_per_inter) == 1:
            v_tuple = mG_per_inter[0]
            h_1 = v_tuple[0]
            h_2 = v_tuple[1]

            estimated_adj_matrix[h_2][h_1] = 1
            estimated_adj_matrix[h_1][h_2] = 1

    return estimated_adj_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_hidden', type=int, default= 3, help='number of hidden variables')
    parser.add_argument('--num_observed', type=int, default= 5, help='number of observeed variables')
    parser.add_argument('--mode', type=str, default= "purechild", help='mode =[purechild, singlesoure] ')
    parser.add_argument('--puregraph', action='store_true', help='compare all invariant models')

    args = parser.parse_args()

    num_samples = 1000

    nb_experiments = 100
    stats_dict = {}
    stats_dict["total_num_exp"] = 0
    stats_dict["bi_failure"] = 0
    stats_dict["failure"] = 0
    stats_dict["metrics_list"] = []

    for _ in tqdm(range(nb_experiments)):
        # print("new experiment")
        stats_dict["total_num_exp"] += 1
        # Create graph
        MMgraph = Latent_and_Bipartite_graph(args.num_hidden, args.num_observed)

        # Find maximal cliques and maximal subsets
        list_maximal_subsets, set_all_maximal_cliques, list_maximal_cliques = find_maximal_cliques(MMgraph, args)

        # Identify the bipartite graph
        biparG = identify_bipartie_graph(list_maximal_subsets, set_all_maximal_cliques, mode=args.mode)
        if len(biparG) > args.num_hidden:
            stats_dict["bi_failure"] += 1
            stats_dict["failure"] += 1
            # print(biparG)
            # exit()
            
            continue
        # else:
        #     print("biparite graph")
        #     print(biparG)

        # print(MMgraph.bipgraph.adj)
        # print(MMgraph.latentdag.adj)
        
        print(biparG)
        # Identify the latent graph
        latentG = identify_latent_graph(biparG, list_maximal_cliques)
        # print("latent graph")
        print(latentG)

        # Evaluation
        # step 0: create estimated_biadj
        nb_estimated_hidden = len(biparG)
        estimated_biadj = np.zeros((args.num_observed, nb_estimated_hidden))
        for i, sets_of_observed in enumerate(biparG):
            for j in sets_of_observed:
                estimated_biadj[j, i] = 1

        # Step 1: map
        true_biadj = MMgraph.bipgraph.adj
        mapping = None
        for perm in itertools.permutations(list(range(nb_estimated_hidden))):
            match = True
            for i in range(nb_estimated_hidden):
                    for j in range(args.num_observed):
                        if true_biadj[j,perm[i]] != estimated_biadj[j, i]:
                            match = False
                            break

            if match:
                    mapping = list(perm)
                    break
        # Step 2: calculate the metrics
        if not mapping:
            final_metrics = {}
            mapping = list(range(nb_estimated_hidden))
            for perm in itertools.permutations(list(range(nb_estimated_hidden))):
                mapping = list(perm)
                metrics = get_metrics(MMgraph, latentG, biparG, mapping)
                if not final_metrics:
                    final_metrics = metrics
                elif final_metrics[0]["shd"] > metrics[0]["shd"]:
                    final_metrics = metrics
        else:
            final_metrics = get_metrics(MMgraph, latentG, biparG, mapping)

        stats_dict["metrics_list"].append(final_metrics)
        print(final_metrics)
        # if final_metrics[2]["undirected_extra"] > 0 or final_metrics[1]["undirected_extra"] > 0:
        if final_metrics[2]["undirected_extra"] > 0:
            exit()

        # print(biparG)
        # print(final_metrics[1])
    # calculate failure rate
    print(args.num_hidden, args.num_observed)


    # shd_list = []
    # nb_edges_list = []
    # extra_list = []
    # missing_list = []
    # for metrics in stats_dict["metrics_list"]:
    #     shd_list.append(metrics[0]["undirected_extra"])
    #     extra_list.append(metrics[0]["undirected_missing"])
    #     missing_list.append(metrics[0]["shd"])
    #     nb_edges_list.append(metrics[0]["total_edges"])


    # print(stats_dict["bi_failure"]/stats_dict["total_num_exp"], stats_dict["failure"]/stats_dict["total_num_exp"], np.average(np.array(shd_list)), np.average(np.array(extra_list)),np.average(np.array(missing_list)), np.average(np.array(nb_edges_list)))

    print(stats_dict["bi_failure"]/stats_dict["total_num_exp"], stats_dict["failure"]/stats_dict["total_num_exp"])

    exp_name = ["full", "biparite", "latent"]
    for exp in [0, 1, 2]:
        print(exp_name[exp])
        
        for metric_key in ["undirected_extra", "undirected_missing", "reverse", "shd", "total_edges"]:
            values_list = []
            for metrics in stats_dict["metrics_list"]:
                values_list.append(metrics[exp][metric_key])
        
            print(metric_key + ":" + str(np.average(np.array(values_list))), end =" ")

        print("\n")

        