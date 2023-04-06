import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def make_graph(returns):
    G = nx.MultiDiGraph()
    G.add_weighted_edges_from(
        [
            (agent, partner, float(sum(ret)))
            for agent, agent_returns in enumerate(returns)
            for partner, ret in enumerate(agent_returns)
            if sum(ret) > 0
        ]
    )
    return G


def make_frequencies_graph(returns):
    G = nx.MultiDiGraph()
    G.add_weighted_edges_from(
        [
            (agent, partner, len(ret))
            for agent, agent_returns in enumerate(returns)
            for partner, ret in enumerate(agent_returns)
            if len(ret) > 0
        ]
    )
    return G


def plot_graph(G, title: str = None):
    plt.figure()
    pos = nx.circular_layout(G)

    colors = []
    weights = []
    for (u, v, attrib_dict) in list(G.edges.data()):
        weights.append(attrib_dict["weight"])
        if u <= v:
            colors.append("blue")
        else:
            colors.append("green")

    max_weight = max(weights)
    weights = [weight / max_weight * 16 for weight in weights]

    if title:
        ax = plt.gca()
        ax.set_title(title)

    nx.draw(
        G,
        pos,
        font_color="white",
        node_shape="s",
        with_labels=True,
        width=weights,
        edge_color=colors,
        connectionstyle="arc3, rad = 0.2",
        # arrowstyle="Simple",
    )


def plot_graph_spring(G, title: str = None):
    plt.figure()
    pos = nx.spring_layout(G, k=2 / (G.order() ** 0.5))

    colors = []
    weights = []
    for (u, v, attrib_dict) in list(G.edges.data()):
        weights.append(attrib_dict["weight"])
        if u <= v:
            colors.append("blue")
        else:
            colors.append("green")

    max_weight = max(weights)
    weights = [weight / max_weight * 10 for weight in weights]

    if title:
        ax = plt.gca()
        ax.set_title(title)

    nx.draw(
        G,
        pos,
        font_color="white",
        node_shape="s",
        with_labels=True,
        width=weights,
        edge_color=colors,
        connectionstyle="arc3, rad = 0.2",
    )


def plot_matrix(matrix, digits=0, figsize=(18, 10)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    m = ax.matshow(matrix)
    fig.colorbar(m)

    for (i, j), z in np.ndenumerate(matrix):
        ax.text(
            j, i, round(z, digits), ha="center", va="center", color="white"
        )  # bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3')

    # return fig


def print_returns(returns):
    for agent in returns:
        print(f"{str(agent) + ':':<20}{sum(returns[agent].values())}")


def plot_degree_distribution(G, t=None):
    degree_sequence = sorted((d for _, d in G.degree()), reverse=True)

    title = "Graph degree"
    if t is not None:
        title += f" (threshold={t})"

    fig = plt.figure(title, figsize=(8, 8))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 4)

    ax0 = fig.add_subplot(axgrid[0:3, :])
    Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    pos = nx.spring_layout(Gcc, seed=10396953)
    nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
    nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
    ax0.set_title(f"Connected components of G (threshold={t})")
    ax0.set_axis_off()

    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Degree Rank Plot")
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Rank")

    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Degree Histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("# of Nodes")

    fig.tight_layout()
    plt.show()
