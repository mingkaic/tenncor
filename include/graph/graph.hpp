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

#include <list>
#include <unordered_map>

#include "include/utils/utils.hpp"

#include "proto/serial/graph.pb.h"

#pragma once
#ifndef TENNCOR_GRAPH_HPP
#define TENNCOR_GRAPH_HPP

namespace nnet
{

class inode;

class varptr;

class variable;

using LEAF_SET = std::unordered_set<std::shared_ptr<inode> >;

using ROOT_STR = std::unordered_set<std::string>;

#define NODE_TYPE tenncor::node_proto::node_t
#define PLACEHOLDER_T tenncor::node_proto::PLACEHOLDER
#define CONSTANT_T tenncor::node_proto::CONSTANT
#define VARIABLE_T tenncor::node_proto::VARIABLE
#define FUNCTOR_T tenncor::node_proto::FUNCTOR

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



	// >>>>>>>>>>>> GRAPH INFO <<<<<<<<<<<<

	bool has_node (inode* node) const;

	inode* get_inst (std::string uid) const;



	// >>>>>>>>>>>> SERIALIZATION <<<<<<<<<<<<

	// serialize entire graph structure
	void serialize (tenncor::graph_proto& proto_dest) const;

	// generate graph from proto
	// set leaves and root in respective out sets
	void register_proto (LEAF_SET& leafset, ROOT_STR& rootstrs,
		const tenncor::graph_proto& proto_src);

	// serialize data to proto_dest based on current graph position
	bool save_data (tenncor::repository_proto& proto_dest) const;

	// load data from proto_src to current graph structure
	void load_data (const tenncor::repository_proto& proto_src);

protected:
	std::string register_node (inode* node);

	void unregister_node (inode* node);

	friend class inode;

private:
	graph (void) {}

	std::string gid_ = nnutils::uuid(this);

	std::unordered_map<std::string,variable*> data_eps_;

	using iter = std::list<inode*>::iterator;

	std::unordered_map<std::string,iter> adjmap_;

	// creation order implies dependency order. independents are created first
	std::list<inode*> order_;
};

}

#endif /* TENNCOR_GRAPH_HPP */
