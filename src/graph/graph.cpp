//
//  graph.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2018-01-12.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/graph/func/functor.hpp"
#include "include/graph/leaf/variable.hpp"
#include "include/graph/leaf/placeholder.hpp"

#ifdef TENNCOR_GRAPH_HPP

namespace nnet
{

bool graph::has_node (inode* node) const
{
	return adjmap_.end() != adjmap_.find(node->get_uid());
}

inode* graph::get_inst (std::string uid) const
{
	auto it = adjmap_.find(uid);
	if (adjmap_.end() == it)
	{
		return nullptr;
	}
	return it->second.first;
}

void graph::serialize (tenncor::graph_proto& proto_dest) const
{
	proto_dest.clear_node_map();
	proto_dest.clear_create_order();
	// set graph_proto node_map (1)
	google::protobuf::Map<std::string,tenncor::node_proto>& nmap = *(proto_dest.mutable_node_map());
	for (auto it = adjmap_.begin(), et = adjmap_.end(); it != et; it++)
	{
		std::string uid = it->first;
		adjiter adjpair = it->second;
		inode* node_src = adjpair.first;
		NODE_TYPE nodetype = node_src->node_type();
		tenncor::node_proto& node_dest = nmap[uid];

		// set node_proto type (1)
		node_dest.set_type(nodetype);
		// set node_proto label (2)
		node_dest.set_label(node_src->get_label());
		// set node_proto detail (3)
		node_src->serialize_detail(node_dest.mutable_detail());
	}
	// set graph_proto create_order (2)
	for (const std::string& ord : order_)
	{
		proto_dest.add_create_order(ord);
	}
}

void graph::register_proto (LEAF_SET& leafset, ROOT_STR& rootstrs,
	const tenncor::graph_proto& proto_src)
{
	// clear everything
	adjmap_.clear();
	order_.clear();
	const google::protobuf::Map<std::string,tenncor::node_proto>& nmap = proto_src.node_map();
	const google::protobuf::RepeatedPtrField<std::string>& order = proto_src.create_order();
	for (auto it = order.begin(), et = order.end(); it != et; it++)
	{
		std::string uid = *it;
		const tenncor::node_proto& node_src = nmap.at(uid);
		inode* node_dest;
		NODE_TYPE nodetype = node_src.type();
		std::string label = node_src.label();
		switch (nodetype)
		{
			case VARIABLE_T:
			{
				tenncor::variable_proto var_src;
				node_src.detail().UnpackTo(&var_src);
				node_dest = new variable(var_src, label, uid);
			}
			break;
			case PLACEHOLDER_T:
			{
				tenncor::shape_proto shape_src;
				node_src.detail().UnpackTo(&shape_src);
				node_dest = new placeholder(shape_src, label, uid);
			}
			break;
			case CONSTANT_T:
			{
				tenncor::tensor_proto tens_src;
				node_src.detail().UnpackTo(&tens_src);
				node_dest = new constant(tens_src, label, uid);
			}
			break;
			case FUNCTOR_T:
			{
				tenncor::functor_proto functor_src;
				node_src.detail().UnpackTo(&functor_src);
				node_dest = new functor(functor_src, label, uid);

				auto strs = functor_src.args();
				// remove args from rootstrs
				for (std::string astr : strs)
				{
					rootstrs.erase(astr);
				}
			}
			break;
			default:
				throw std::exception(); // unsupported node implementation
		}
		// register node
		auto oit = order_.insert(order_.end(), uid);
		adjmap_[uid] = {node_dest, oit};
		// add to leaf set
		if (nodetype != FUNCTOR_T)
		{
			leafset.emplace(std::shared_ptr<inode>(node_dest));
		}
		// add to rootstrs
		rootstrs.emplace(uid);
	}
}

std::string graph::register_node (inode* node)
{
	std::string uid = nnutils::uuid(node);
	auto it = order_.insert(order_.end(), uid);
	adjmap_[uid] = {node, it};
	return uid;
}

void graph::unregister_node (inode* node)
{
	auto it = adjmap_.find(node->get_uid());
	if (adjmap_.end() != it)
	{
		order_.erase(it->second.second);
		adjmap_.erase(it);
	}
}

}

#endif
