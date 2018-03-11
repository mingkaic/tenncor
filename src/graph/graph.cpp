//
//  graph.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2018-01-12.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/graph/inode.hpp"

#ifdef TENNCOR_GRAPH_HPP

namespace nnet
{

bool graph::has_node (inode* node) const
{
	return adjlist_.end() != adjlist_.find(node->get_uid());
}

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
