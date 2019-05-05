# not doing this locally in C++ cos the graphviz API is nightmarish

from datetime import datetime
import time
import random

import grpc
from concurrent import futures

import ead.dbg.graph_pb2 as graph_pb2
import ead.dbg.graph_pb2_grpc as graph_pb2_grpc
import ead.dbg.interactive_renderer as renderer

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

colors = [
    'antiquewhite4', 'aquamarine4', 'azure4',
    'black', 'blanchedalmond', 'blue',
    'blueviolet', 'brown', 'burlywood',
    'cadetblue', 'chartreuse', 'chocolate',
    'coral', 'cornflowerblue', 'cornsilk3',
    'crimson', 'cyan', 'darkgoldenrod',
    'darkgreen', 'darkkhaki', 'darkolivegreen',
    'darkorange', 'darkorchid', 'darksalmon',
    'darkseagreen', 'darkslateblue', 'darkslategray',
    'darkslategrey', 'darkturquoise', 'darkviolet',
    'deeppink', 'deepskyblue', 'dimgray',
    'dimgrey', 'dodgerblue', 'firebrick',
    'forestgreen', 'gold', 'goldenrod',
    'gray', 'green', 'greenyellow',
    'grey', 'honeydew3', 'hotpink',
    'indianred', 'indigo', 'invis',
    'ivory3', 'khaki', 'lavender',
    'lavenderblush', 'lawngreen', 'lemonchiffon4',
    'lightblue2', 'lightcoral', 'lightcyan4',
    'lightgrey', 'lightpink', 'lightsalmon',
    'lightseagreen', 'lightskyblue', 'lightslateblue',
    'lightslategray', 'lightslategrey', 'lightsteelblue',
    'lightyellow4', 'limegreen', 'paleturquoise4',
    'magenta', 'maroon', 'mediumaquamarine',
    'mediumblue', 'mediumorchid', 'mediumpurple',
    'mediumseagreen', 'mediumslateblue', 'mediumspringgreen',
    'mediumturquoise', 'mediumvioletred', 'midnightblue',
    'navajowhite', 'navy', 'navyblue',
    'orange', 'orangered', 'orchid',
    'papayawhip', 'peachpuff', 'peru',
    'pink', 'plum', 'purple',
    'rosybrown', 'royalblue', 'saddlebrown',
    'salmon', 'sandybrown', 'seagreen',
    'seashell4', 'sienna', 'skyblue',
    'slateblue', 'slategray', 'slategrey',
    'springgreen', 'steelblue', 'red',
    'tan', 'thistle', 'tomato',
    'turquoise', 'violet', 'violetred',
    'wheat', 'yellow', 'yellowgreen',
    'mistyrose', 'olivedrab', 'palegreen',
    'lightgoldenrod', 'lightgray', 'palevioletred',
]

random.seed(datetime.now())
random.shuffle(colors)

def transport_to_graph(graph):
    nodes = graph.nodes
    edges = graph.edges

    out_node = []
    out_edge = {}

    label_map = {}
    node_map = {}
    for node in nodes:
        node_repr = str(node.id) + '=' + str(node.repr)
        node_map[node.id] = node_repr
        labels = [str(label) for label in node.labels]
        labels.sort()
        label = ','.join(labels)
        if label in label_map:
            color = label_map[label]
        else:
            color = colors[len(label_map)]
            label_map[label] = color
        out_node.append((node_repr, color))

    for edge in edges:
        parent = node_map[edge.parent]
        child_tup = (node_map[edge.child], edge.label)
        if parent in out_edge:
            out_edge[parent].append(child_tup)
        else:
            out_edge[parent] = [child_tup]

    return out_node, out_edge, label_map

class InteractiveGrapherServicer(graph_pb2_grpc.InteractiveGrapherServicer):
    def UpdateGraph(self, request, context):
        graph = request.payload

        renderer.print_graph(transport_to_graph(graph), '/tmp/interactive_graph')

        return graph_pb2.GraphUpdateResponse(
            status=graph_pb2.GraphUpdateResponse.OK,
            message='OK'
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    graph_pb2_grpc.add_InteractiveGrapherServicer_to_server(
        InteractiveGrapherServicer(), server)
    server.add_insecure_port('localhost:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
