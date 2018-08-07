#include <list>

#include "glass/session.hpp"
#include "glass/data.pb.h"

#ifndef GLASS_NODE_HPP
#define GLASS_NODE_HPP

void save_info (tenncor::DataInfoPb& out, iNode* data);

DTYPE load_info (Shape& out, const tenncor::DataInfoPb& info);

void save_node (tenncor::DataPb& out, iNode* data);

DTYPE load_node (std::string& out, Shape& outshape, const tenncor::DataPb& data);


// extract named nodes in session
// then breadth first traverse all nodes in graph
// deterministicallt establish order for adjacency list indexing
std::list<iNode*> order_nodes (const Session& in);

// same as order_nodes above except map nodes to index
// when node and index need to be bimapped
std::list<iNode*> order_nodes (const Session& in,
    std::unordered_map<iNode*,uint32_t>& nodemap);

#endif /* GLASS_NODE_HPP */
