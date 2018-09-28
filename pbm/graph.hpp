/*!
 *
 *  graph.hpp
 *  pbm
 *
 *  Purpose:
 *  define functions for marshal and unmarshal operation graph
 *
 */

#include "llo/node.hpp"

#include "pbm/graph.pb.h"

/*! marshal all operation subgraphs in root vector to protobuf object */
void save_graph (tenncor::Graph& out, std::vector<ade::Tensorptr>& roots);

/*! unmarshal protobuf object and return all nodes in graph */
std::vector<ade::Tensorptr> load_graph (const tenncor::Graph& in);
