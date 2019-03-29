# not doing this locally in C++ cos the graphviz API is nightmarish

import time

import grpc
from concurrent import futures

import ead.dbg.graph_pb2 as graph_pb2
import ead.dbg.graph_pb2_grpc as graph_pb2_grpc
import ead.dbg.interactive_renderer as renderer

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

colors = ['chartreuse', 'crimson', 'cyan', 'gold', 'chocolate', 'darkviolet', 'gray', 'aquamarine', 'firebrick', 'dodgerblue']

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
        if node.label in label_map:
            color = label_map[str(node.label)]
        else:
            color = colors[len(label_map)]
            label_map[node.label] = color
        out_node.append((node_repr, color))

    for edge in edges:
        parent = node_map[edge.parent]
        child_tup = (node_map[edge.child], edge.order)
        if parent in out_edge:
            out_edge[parent].append(child_tup)
        else:
            out_edge[parent] = [child_tup]

    return out_node, out_edge

class InteractiveGrapherServicer(graph_pb2_grpc.InteractiveGrapherServicer):
    def UpdateGraph(self, request, context):
        graph = request.payload

        renderer.print_graph(transport_to_graph(graph), 'interactive_graph')

        return graph_pb2.GraphUpdateResponse(
            status=graph_pb2.GraphUpdateResponse.OK,
            message="OK"
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
