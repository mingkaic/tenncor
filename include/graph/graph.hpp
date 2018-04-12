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
#include "include/tensor/data_src.hpp"

#include "proto/serial/graph.pb.h"

#pragma once
#ifndef TENNCOR_GRAPH_HPP
#define TENNCOR_GRAPH_HPP

namespace nnet
{

class inode;

class varptr;

class variable;

struct vphash
{
	size_t operator() (const varptr& vp) const;
};

using LEAF_SET = std::unordered_set<varptr, vphash>;

using ROOT_STR = std::unordered_set<std::string>;

using NODE_TYPE = tenncor::NodePb::NodeT;

static const NODE_TYPE PLACEHOLDER_T = tenncor::NodePb::PLACEHOLDER;
static const NODE_TYPE CONSTANT_T = tenncor::NodePb::CONSTANT;
static const NODE_TYPE VARIABLE_T = tenncor::NodePb::VARIABLE;
static const NODE_TYPE FUNCTOR_T = tenncor::NodePb::FUNCTOR;

class graph
{
public:
	static graph& get_global (void)
	{
		static graph g;
		return g;
	}

	static std::unique_ptr<graph> get_temp (void)
	{
		return std::unique_ptr<graph>(new graph());
	}

	static void replace_global (std::unique_ptr<graph>&& temp)
	{
		get_global() = std::move(*temp);
	}

	graph (const graph&) = delete;
	graph (graph&&) = delete;
	graph& operator = (const graph&) = delete;



	// >>>>>>>>>>>> GRAPH INFO <<<<<<<<<<<<

	bool has_node (inode* node) const;

	bool has_node (std::string uid) const;

	inode* get_inst (std::string uid) const;

	std::string get_gid (void) const
	{
		return gid_;
	}



	// >>>>>>>>>>>> SERIALIZATION <<<<<<<<<<<<

	// serialize entire graph structure
	void serialize (tenncor::GraphPb& proto_dest) const;

	// generate graph from proto
	// set leaves and root in respective out sets
	// overwrites existing graph
	void register_proto (LEAF_SET& leafset, ROOT_STR& rootstrs,
		const tenncor::GraphPb& proto_src);

	// serialize data to proto_dest based on current graph position
	bool save_data (tenncor::DataRepoPb& proto_dest) const;

	// load data from proto_src to current graph structure
	void load_data (const tenncor::DataRepoPb& proto_src);

protected:
	std::string register_node (inode* node);

	void unregister_node (inode* node);

	friend class inode;

private:
	graph (void) = default;

	graph& operator = (graph&&) = default;

	std::string gid_ = nnutils::uuid(this);

	using iter = std::list<inode*>::iterator;

	std::unordered_map<std::string,iter> adjmap_;

	// creation order implies dependency order. independents are created first
	std::list<inode*> order_;
};

}

#endif /* TENNCOR_GRAPH_HPP */
