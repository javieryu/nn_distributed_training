import networkx as nx


def generate_from_conf(graph_conf):
    """ Generates a graph given a specified configuration.

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