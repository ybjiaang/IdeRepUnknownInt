import numpy as np

def count_accuracy(B_true, B_est):
    """
    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG
    # SHD = undirected extra (skeleton) + undirected missing (skeleton) + reverse (directed graph)
    # unoriented_correct = # undirected edges in the cpdag that has a corresponding true edge in the true dag
    """
    d = len(B_true)
    assert (len(B_est) == d)
    undirected_extra = 0
    undirected_missing = 0
    reverse = 0
    unoriented_correct = 0
    total_edges = 0
    for i in range(d):
        for j in range(i + 1, d):
            undir_true = (B_true[i][j] == 1 or B_true[j][i] == 1)
            undir_est = (B_est[i][j] == 1 or B_est[i][j] == -1 or B_est[j][i] == 1 or B_est[j][i] == -1)
            if undir_true:
                total_edges += 1
            if undir_true and (not undir_est):
                undirected_missing += 1
            elif (not undir_true) and undir_est:
                undirected_extra += 1
            elif undir_true and undir_est:
                if B_est[i][j] == -1 or B_est[j][i] == -1:
                    # Undirected edge in est
                    unoriented_correct += 1
                elif B_true[i][j] != B_est[i][j]:
                    # Directed edge in est, but reversed
                    reverse += 1
    fdr = 0
    if total_edges > 0:
        fdr = reverse/total_edges
    return {"shd": undirected_extra + undirected_missing + reverse,
            "undirected_extra": undirected_extra,
            "undirected_missing": undirected_missing,
            "reverse": reverse,
            "unoriented_correct": unoriented_correct,
            "total_edges": total_edges,
            "fdr": fdr}

def get_metrics(MMgraph, latentG, biparG, mapping):
    num_hidden, num_observed = MMgraph.num_hidden, MMgraph.num_observed
    num_estimated_hidden = len(biparG)
    num_max_hidden = max(num_hidden, num_estimated_hidden)

    B_true = [[0 for i in range(num_max_hidden + num_observed)] for j in range(num_max_hidden + num_observed)]
    B_bi_true = [[0 for i in range(num_max_hidden + num_observed)] for j in range(num_max_hidden + num_observed)]
    B_latent_true = [[0 for i in range(num_max_hidden + num_observed)] for j in range(num_max_hidden + num_observed)]

    for u in range(num_hidden):
        for v in range(num_hidden):
            if MMgraph.latentdag.adj[v, u] == 1:
                B_true[v][u] = 1
                B_latent_true[v][u] = 1

    for u in range(num_hidden):
        for v in range(num_observed):
            if MMgraph.bipgraph.adj[v, u] == 1:
                B_true[num_hidden + v][u] = 1
                B_bi_true[num_hidden + v][u] = 1

    B_est = [[0 for i in range(num_max_hidden + num_observed)] for j in range(num_max_hidden + num_observed)]
    B_bi_est = [[0 for i in range(num_max_hidden + num_observed)] for j in range(num_max_hidden + num_observed)]
    B_latent_est = [[0 for i in range(num_max_hidden + num_observed)] for j in range(num_max_hidden + num_observed)]
    nb_estimated_hidden = len(biparG)
    for u in range(nb_estimated_hidden):
        for v in range(u + 1, nb_estimated_hidden):
            if latentG[u][v] == 1 and latentG[v][u] == 1:
                B_est[mapping[u]][mapping[v]] = -1
                B_latent_est[mapping[u]][mapping[v]] = -1
            elif latentG[u][v] == 1:
                B_est[mapping[u]][mapping[v]] = 1
                B_latent_est[mapping[u]][mapping[v]] = 1
            elif latentG[v][u] == 1:
                B_est[mapping[v]][mapping[u]] = 1
                B_latent_est[mapping[v]][mapping[u]] = 1

    for u in range(nb_estimated_hidden):
        for v in range(num_observed):
            if v in biparG[u]:
                B_est[num_hidden + v][mapping[u]] = 1
                B_bi_est[num_hidden + v][mapping[u]] = 1

    full_metrics = count_accuracy(B_true, B_est)
    bi_metrics = count_accuracy(B_bi_true, B_bi_est)
    latent_metrics = count_accuracy(B_latent_true, B_latent_est)
    return (full_metrics, bi_metrics, latent_metrics)
            
