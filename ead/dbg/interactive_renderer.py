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

def _str_clean(label):
    # get rid of <, >, |, :
    label = label.strip()\
        .replace('<', '(')\
        .replace('>', ')')\
        .replace('|', '!')\
        .replace('\\', ',')\
        .replace('::', '.')\
        .replace(':', '=')\
        .replace(',,', '\\n')
    print(label)
    return label

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
    nodes, edges, legend = callgraph

    g1 = gv.Digraph(format='png')
    for node, color in nodes:
        node = _str_clean(node)
        g1.node(node, fillcolor=color)

    for observer in edges:
        for subject, order in edges[observer]:
            observer = _str_clean(observer)
            subject = _str_clean(subject)
            label = str(order)
            g1.edge(observer, subject, label)

    labels = legend.keys()
    labels.sort(lambda x, y: len(x) - len(y))
    for label in labels:
        color = legend[label]
        g1.node(_str_clean(label), fillcolor=color, shape='box')
    for label1, label2 in zip(labels[:-1], labels[1:]):
        g1.edge(_str_clean(label1), _str_clean(label2),
            color='transparent')

    _apply_styles(g1, _styles)
    g1.render(outname, view=True)
