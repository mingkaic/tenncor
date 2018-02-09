//
//  graph.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2018-01-12.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/graph/connector/iconnector.hpp"

#ifdef TENNCOR_GRAPH_HPP

namespace nnet
{

std::string graph::register_node (inode* node)
{
	std::string uid = nnutils::uuid(node);
	adjlist_[uid] = node;
	return uid;
}

void graph::unregister_node (inode* node)
{
	auto it = adjlist_.find(node->get_uid());
	if (adjlist_.end() != it)
	{
		adjlist_.erase(it);
	}
}

}

#endif
