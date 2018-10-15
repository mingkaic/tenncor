///
/// graph.hpp
/// pbm
///
/// Purpose:
/// Define functions for marshal and unmarshal equation graph
///

#include "llo/node.hpp"

#include "pbm/graph.pb.h"

/// Marshal all equation graphs in roots vector to protobuf object
void save_graph (tenncor::Graph& out, std::vector<llo::DataNode>& roots);

/// Return all nodes in graph unmarshalled from protobuf object
std::vector<llo::DataNode> load_graph (const tenncor::Graph& in);
