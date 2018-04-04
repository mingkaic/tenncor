//
//  graph.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2018-01-12.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/operate/operations.hpp"
#include "include/graph/variable.hpp"
#include "include/graph/placeholder.hpp"

#ifdef TENNCOR_GRAPH_HPP

namespace nnet
{

bool graph::has_node (inode* node) const
{
	return adjmap_.end() != adjmap_.find(node->get_uid());
}

bool graph::has_node (std::string uid) const
{
	return adjmap_.end() != adjmap_.find(uid);
}

inode* graph::get_inst (std::string uid) const
{
	auto it = adjmap_.find(uid);
	if (adjmap_.end() == it)
	{
		return nullptr;
	}
	return *(it->second);
}

void graph::serialize (tenncor::graph_proto& proto_dest) const
{
	proto_dest.set_gid(gid_);
	proto_dest.clear_node_map();
	proto_dest.clear_create_order();
	// set graph_proto node_map (1)
	google::protobuf::Map<std::string,tenncor::node_proto>& nmap = *(proto_dest.mutable_node_map());
	for (auto it = adjmap_.begin(), et = adjmap_.end(); it != et; it++)
	{
		std::string uid = it->first;
		iter adj = it->second;
		inode* node_src = *adj;
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
	for (inode* const& ord : order_)
	{
		proto_dest.add_create_order(ord->get_uid());
	}
}

static inline variable* make_variable (tenncor::variable_proto var_src, std::string label)
{
	const tenncor::source_proto& source_src = var_src.source();
	nnet::SOURCE_TYPE src_type = source_src.src();
	std::shared_ptr<data_src> src;
	switch (src_type)
	{
		case CSRC_T:
			src = std::make_shared<const_init>(
				source_src.settings(0), source_src.dtype());
		break;
		case USRC_T:
			src = std::make_shared<r_uniform_init>(
				source_src.settings(0), source_src.settings(1),
				source_src.dtype());
		break;
		case NSRC_T:
			src = std::make_shared<r_normal_init>(
				source_src.settings(0), source_src.settings(1),
				source_src.dtype());
		break;
		default:
			throw std::exception(); // unsupported data source
	}

	const google::protobuf::RepeatedField<uint64_t>& shape_src = var_src.allowed_shape();
	std::vector<size_t> shape(shape_src.begin(), shape_src.end());
	return new variable(shape, src, label);
}

static inline placeholder* make_placeholder (tenncor::place_proto& place_src, std::string label)
{
	const google::protobuf::RepeatedField<uint64_t>& shape_src = place_src.allowed_shape();
	std::vector<size_t> shape(shape_src.begin(), shape_src.end());
	return new placeholder(shape, label);
}

void graph::register_proto (LEAF_SET& leafset, ROOT_STR& rootstrs,
	const tenncor::graph_proto& proto_src)
{
	// todo: support multiple graphs (return new graph)
	gid_ = proto_src.gid();
	// clear everything
	adjmap_.clear();
	order_.clear();
	const google::protobuf::Map<std::string,tenncor::node_proto>& nmap = proto_src.node_map();
	const google::protobuf::RepeatedPtrField<std::string>& order = proto_src.create_order();
	std::unordered_map<std::string, std::string> uid_map;
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
				variable* var = make_variable(var_src, label);
				var->varpos_ = var_src.varpos();
				data_eps_.erase(var->get_uid());
				data_eps_[var->varpos_] = var;
				node_dest = var;
			}
			break;
			case PLACEHOLDER_T:
			{
				tenncor::place_proto place_src;
				node_src.detail().UnpackTo(&place_src);
				node_dest = make_placeholder(place_src, label);
			}
			break;
			case CONSTANT_T:
			{
				tenncor::tensor_proto tens_src;
				node_src.detail().UnpackTo(&tens_src);
				node_dest = constant::get(tens_src, label);
			}
			break;
			case FUNCTOR_T:
			{
				tenncor::functor_proto functor_src;
				node_src.detail().UnpackTo(&functor_src);
				auto strs = functor_src.args();
				std::vector<inode*> args(strs.size());
				std::transform(strs.begin(), strs.end(), args.begin(),
				[&](std::string src) -> inode*
				{
					std::string stored_uid = uid_map[src];
					rootstrs.erase(stored_uid);
					return this->get_inst(stored_uid);
				});
				node_dest = run_opcode(args, (OPCODE) functor_src.opcode());
				node_dest->set_label(label);
			}
			break;
			default:
				throw std::exception(); // unsupported node implementation
		}
		// uid map
		std::string resuid = node_dest->get_uid();
		uid_map[uid] = resuid;
		// register node
		auto oit = order_.insert(order_.end(), node_dest);
		adjmap_[uid] = oit;
		// add to leaf set
		if (nodetype != FUNCTOR_T)
		{
			leafset.emplace(std::shared_ptr<inode>(node_dest));
		}
		// add to rootstrs
		rootstrs.emplace(resuid);
	}
}

bool graph::save_data (tenncor::repository_proto& proto_dest) const
{
	auto dmap = proto_dest.mutable_data_map();
	for (auto eps : data_eps_)
	{
		tenncor::tensor_proto tens_dest;
		if (false == eps.second->get_tensor()->serialize(tens_dest))
		{
			return false;
		}
		dmap->insert({eps.first, tens_dest});
	}
	return true;
}

void graph::load_data (const tenncor::repository_proto& proto_src)
{
	auto dmap = proto_src.data_map();
	for (auto dpair : dmap)
	{
		std::string pos = dpair.first;
		auto it = data_eps_.find(pos);
		if (data_eps_.end() != it)
		{
			tensor* tens = it->second->get_tensor();
			tens->from_proto(dpair.second);
		}
	}
}

std::string graph::register_node (inode* node)
{
	std::string uid = nnutils::uuid(node);
	auto it = order_.insert(order_.end(), node);
	adjmap_[uid] = it;
	if (variable* var = dynamic_cast<variable*>(node))
	{
		data_eps_[var->get_uid()] = var;
	}
	return uid;
}

void graph::unregister_node (inode* node)
{
	auto it = adjmap_.find(node->get_uid());
	if (adjmap_.end() != it)
	{
		order_.erase(it->second);
		adjmap_.erase(it);
		if (variable* var = dynamic_cast<variable*>(node))
		{
			data_eps_.erase(var->varpos_);
		}
	}
}

}

#endif
