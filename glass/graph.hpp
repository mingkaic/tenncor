#include "glass/node.hpp"
#include "glass/graph.pb.h"

#ifndef GLASS_GRAPH_HPP
#define GLASS_GRAPH_HPP

void save_graph (tenncor::GraphPb& out, const Session& in);

std::vector<Nodeptr> load_graph (Session& out, const tenncor::GraphPb& in);

#endif /* GLASS_GRAPH_HPP */
