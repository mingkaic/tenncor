/*!
 *
 *  graph.hpp
 *  cnnet
 *
 *  Purpose:
 *  graph adjlist
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include <unordered_map>

#include "include/utils/utils.hpp"

#pragma once
#ifndef TENNCOR_GRAPH_HPP
#define TENNCOR_GRAPH_HPP

namespace nnet
{

class inode;

class graph
{
public:
	static graph& get_global (void)
	{
		static graph g;
		return g;
	}

	graph (const graph&) = delete;
	graph (graph&&) = delete;
	graph& operator = (const graph&) = delete;
	graph& operator = (graph&&) = delete;

	bool has_node (inode* node) const;

	// serialize
	bool serialize (tenncor::graph_proto* proto_dest) const
	{
		return false;
	}

	// read from proto
	void read_from (const tenncor::graph_proto& proto_src)
	{
		
	}

protected:
	std::string register_node (inode* node);

	void unregister_node (inode* node);

	friend class inode;

private:
	graph (void) {}

	//! uniquely identifier for this node
	std::string gid_ = nnutils::uuid(this);

	std::unordered_map<std::string,nnet::inode*> adjlist_;
};

}

#endif /* TENNCOR_GRAPH_HPP */
