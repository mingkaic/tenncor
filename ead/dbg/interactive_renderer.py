import sys
from collections import defaultdict

import graphviz as gv

_styles = {
    'graph': {
        'label': 'Operation Graph',
        'fontsize': '16',
        'fontcolor': 'white',
        'bgcolor': '#333333',
        'rankdir': 'BT',
    },
    'nodes': {
        'fontname': 'Helvetica',
        'shape': 'hexagon',
        'fontcolor': 'white',
        'color': 'white',
        'style': 'filled',
        'fillcolor': '#006699',
    },
    'edges': {
        'style': 'dashed',
        'color': 'white',
        'arrowhead': 'open',
        'fontname': 'Courier',
        'fontsize': '12',
        'fontcolor': 'white',
    }
}

def _apply_styles(graph, styles):
    graph.graph_attr.update(
        ('graph' in styles and styles['graph']) or {}
    )
    graph.node_attr.update(
        ('nodes' in styles and styles['nodes']) or {}
    )
    graph.edge_attr.update(
        ('edges' in styles and styles['edges']) or {}
    )
    return graph

def print_graph(callgraph, outname):
    nodes, edges = callgraph

    g1 = gv.Digraph(format='png')
    for node, color in nodes:
        g1.node(node, color=color)

    for observer in edges:
        for subject, order in edges[observer]:
            label = str(order)
            g1.edge(observer, subject, label)

    _apply_styles(g1, _styles)
    g1.render(outname, view=True)
