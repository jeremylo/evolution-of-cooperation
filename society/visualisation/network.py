import matplotlib.pyplot as plt
import networkx as nx


def make_graph(returns):
    G = nx.MultiDiGraph()
    G.add_weighted_edges_from([
        (agent, partner, sum(ret))
        for agent, agent_returns in enumerate(returns)
        for partner, ret in enumerate(agent_returns)
        if sum(ret) > 0
    ])
    return G

def plot_returns(G):
    plt.figure()
    pos = nx.circular_layout(G)

    colors = []
    weights = []
    for (u, v, attrib_dict) in list(G.edges.data()):
        weights.append(attrib_dict['weight'])
        if u <= v:
            colors.append("blue")
        else:
            colors.append("green")

    max_weight = max(weights)
    weights = [weight / max_weight * 20 for weight in weights]

    nx.draw(G, pos, font_color="white", node_shape="s", with_labels=True, width=weights, edge_color=colors, connectionstyle="arc3, rad = 0.2", arrowstyle="Simple")

def print_returns(returns):
    for agent in returns:
        print(f"{str(agent) + ':':<20}{sum(returns[agent].values())}")
