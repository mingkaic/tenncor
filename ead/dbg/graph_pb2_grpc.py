# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import graph_pb2 as graph__pb2


class InteractiveGrapherStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.UpdateGraph = channel.unary_unary(
        '/idbg.InteractiveGrapher/UpdateGraph',
        request_serializer=graph__pb2.GraphUpdateRequest.SerializeToString,
        response_deserializer=graph__pb2.GraphUpdateResponse.FromString,
        )


class InteractiveGrapherServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def UpdateGraph(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_InteractiveGrapherServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'UpdateGraph': grpc.unary_unary_rpc_method_handler(
          servicer.UpdateGraph,
          request_deserializer=graph__pb2.GraphUpdateRequest.FromString,
          response_serializer=graph__pb2.GraphUpdateResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'idbg.InteractiveGrapher', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
