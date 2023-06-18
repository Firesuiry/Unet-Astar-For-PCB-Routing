import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def show_graph(rects, edges):
    g = nx.Graph()
    pos = nx.spring_layout(g)
    for i in range(len(rects)):
        g.add_node(i)
        pos[i] = np.array([int((rects[i]['x0']+rects[i]['x1'])/2), int((rects[i]['y0']+rects[i]['y1'])/2)])
    for rect_ids, limit in edges.items():
        # g.add_edge(rect_ids[0], rect_ids[1], limit=limit)
        g.add_weighted_edges_from([(rect_ids[0], rect_ids[1], limit)])
    fig, ax = plt.subplots(figsize=(40, 20))
    nx.draw(g, pos=pos, ax=ax, with_labels=True, node_size=300, font_size=24)
    plt.show()

    # plt.figure(figsize=(4000, 2000))
    # nx.draw(g, pos=pos, with_labels=True, node_size=100, font_size=8)
    # plt.savefig("graph.png", dpi=250)
    # plt.show()
