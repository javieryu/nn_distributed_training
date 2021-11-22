import networkx as nx
import numpy as np
import scipy
import torch
import scipy.spatial as spatial
import random


def fied_from_disk(N, positions, radius):
    G = nx.random_geometric_graph(N, radius, pos=positions)
    return nx.linalg.algebraic_connectivity(G, tol=1e-3, method="lanczos")


def disk_with_fied(N, targ, num_restarts=50):
    for _ in range(num_restarts):
        pos = {i: (random.random(), random.random()) for i in range(N)}

        # +/- tolerance to target
        tol = 0.01

        # initial bounds
        lbr = 0.05
        ubr = 0.8

        lbf = fied_from_disk(N, pos, lbr)
        ubf = fied_from_disk(N, pos, ubr)

        if abs(lbf - targ) < tol:
            return nx.random_geometric_graph(N, lbr, pos=pos)

        if abs(ubf - targ) < tol:
            return nx.random_geometric_graph(N, ubr, pos=pos)

        if not ubf > lbf:
            print("upper bound fied", ubf)
            print("lower bound fied", lbf)
            raise NameError("Not sure whats happening in graph gen")

        if targ > ubf or targ < lbf:
            print("upper bound fied", ubf)
            print("lower bound fied", lbf)
            raise NameError("Target outside range.")

        flag = True
        c = 0
        while flag:
            midr = (ubr + lbr) / 2

            midf = fied_from_disk(N, pos, midr)

            if abs(midf - targ) < tol:
                return nx.random_geometric_graph(N, midr, pos=pos)

            if midf > targ:
                ubr = midr
                ubf = midf
            elif midf < targ:
                lbr = midr
                lbf = midf

            if c > 100:
                break
            else:
                c += 1

    raise ("Never found a viable graph!")


def generate_from_conf(graph_conf):
    """Generates a graph given a specified configuration.

    Args:
        graph_conf (dict): dictionary containing parameters used
        in the generation of graphs.

    Returns:
        int : the size of the network that was generated
        Graph : a networkx graph representation.
    """
    N = graph_conf["num_nodes"]
    if graph_conf["type"] == "wheel":
        graph = nx.wheel_graph(N)
    elif graph_conf["type"] == "cycle":
        graph = nx.cycle_graph(N)
    elif graph_conf["type"] == "complete":
        graph = nx.complete_graph(N)
    elif graph_conf["type"] == "random":
        # Attempt to make a random graph until it is connected
        graph = nx.erdos_renyi_graph(N, graph_conf["p"])
        for _ in range(graph_conf["gen_attempts"]):
            if nx.is_connected(graph):
                break
            else:
                graph = nx.erdos_renyi_graph(N, graph_conf["p"])

        if not nx.is_connected(graph):
            raise NameError(
                "A connected random graph could not be generated,"
                " increase p or gen_attempts."
            )
    else:
        raise NameError("Unknown communication graph type.")

    return N, graph


def get_metropolis(graph):
    N = graph.number_of_nodes()
    W = torch.zeros((N, N))

    L = nx.laplacian_matrix(graph)
    degs = [L[i, i] for i in range(N)]

    for i in range(N):
        for j in range(N):
            if graph.has_edge(i, j) and i != j:
                W[i, j] = 1.0 / (max(degs[i], degs[j]) + 1.0)

    for i in range(N):
        W[i, i] = 1.0 - torch.sum(W[i, :])

    return W


def euclidean_disk_graph(poses, radius):
    """Takes an array of poses and computes the
    euclidean distances between all poses. The distances
    are thresholded, and a networkx graph is returned

    Args:
        poses (numpy.matrix): poses of nodes shape [N, 2]
        radius (float): nodes with a pairwise distance less
        than this number can communicate.

    Returns:
        networkx.Graph: the generated euclidean disk graph
        bool: the connectivity of the generated graph
    """
    dist_mat = scipy.spatial.distance.pdist(poses, "euclidean")
    dist_mat = scipy.spatial.distance.squareform(dist_mat)
    adj_mat = dist_mat <= radius
    for i in range(adj_mat.shape[0]):
        adj_mat[i, i] = 0
    graph = nx.from_numpy_matrix(adj_mat)

    return graph, nx.is_connected(graph)


def gen_delaunay(N):
    """Generates a graph from the delaunay triangulation
    of N points sampled uniformly from the [0, 1] box.

    Args:
        N (int): Number of nodes

    Returns:
        [networkx.Graph]: Graph of Delaunay triangulation.
    """
    positions = np.random.rand(N, 2)
    tri = spatial.Delaunay(positions)
    edges = []
    for simplex in tri.simplices:
        edges.append((simplex[0], simplex[1]))
        edges.append((simplex[1], simplex[2]))
        edges.append((simplex[0], simplex[2]))

    graph = nx.Graph(edges)
    return graph