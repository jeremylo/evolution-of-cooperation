import matplotlib.pyplot as plt
import networkx as nx


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


def print_returns(returns):
    for agent in returns:
        print(f"{str(agent) + ':':<20}{sum(returns[agent].values())}")
