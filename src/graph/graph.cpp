//
//  graph.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2018-01-12.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/graph/func/functor.hpp"
#include "include/graph/leaf/variable.hpp"

#ifdef TENNCOR_GRAPH_HPP

namespace nnet
{

bool graph::has_node (inode* node) const
{
	return adjlist_.end() != adjlist_.find(node->get_uid());
}

void graph::serialize (tenncor::graph_proto& proto_dest) const
{
	proto_dest.clear_node_map();
	proto_dest.clear_create_order();
	// set graph_proto node_map (1)
	google::protobuf::Map<std::string,tenncor::node_proto>& nmap = *(proto_dest.mutable_node_map());
	for (auto it = adjlist_.begin(), et = adjlist_.end(); it != et; it++)
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
		switch (nodetype)
		{
			case FUNCTOR_T:
			{
				tenncor::functor_proto func_dest;
				std::vector<inode*> args = static_cast<functor*>(node_src)->get_arguments();
				for (inode* arg : args)
				{
					func_dest.add_args(arg->get_uid()); 
				}
				node_dest.mutable_detail()->PackFrom(func_dest);
			}
			break;
			case CONSTANT_T:
			{
				tenncor::tensor_proto tens;
				tensor* data = node_src->get_tensor();
				assert(nullptr != data && data->serialize(tens));
				node_dest.mutable_detail()->PackFrom(tens);
			}
			break;
			case VARIABLE_T:
			case PLACEHOLDER_T:
			tenncor::shape_proto shape;
			{
				tensor* data = node_src->get_tensor();
				std::vector<size_t> allowed = data->get_allowed().as_list();
				google::protobuf::RepeatedField<uint64_t> allowed_field(allowed.begin(), allowed.end());
				shape.mutable_allowed()->Swap(&allowed_field);
				if (data->has_data())
				{
					std::vector<size_t> alloced = data->get_shape().as_list();
					google::protobuf::RepeatedField<uint64_t> alloced_field(alloced.begin(), alloced.end());
					shape.mutable_alloced()->Swap(&alloced_field);
				}
			}
			if (PLACEHOLDER_T == nodetype)
			{
				node_dest.mutable_detail()->PackFrom(shape);
				break;
			}
			{
				variable* var_src = static_cast<variable*>(node_src);
				tenncor::variable_proto var;
				var.mutable_shape()->Swap(&shape);
				
				tenncor::source_proto src_dest;
				var_src->get_source()->serialize(src_dest);
				var.mutable_source()->Swap(&src_dest);

				node_dest.mutable_detail()->PackFrom(var);
			}
			break;
		}
	}
	// set graph_proto create_order (2)
	for (const std::string& ord : order_)
	{
		proto_dest.add_create_order(ord);
	}
}

void graph::register_proto (LEAF_SET& leafset, ROOT_SET& rootset,
	const tenncor::graph_proto& proto_src)
{
	const google::protobuf::Map<std::string,tenncor::node_proto>& nmap = proto_src.node_map();
	const google::protobuf::RepeatedPtrField<std::string>& order = proto_src.create_order();
	
	for (auto it = order.begin(), et = order.end(); it != et; it++)
	{
		const tenncor::node_proto& node_src = nmap.at(*it);
		// switch (node_src.type())
		// {
		// 	case tenncor::node_proto_node_t::node_proto_node_t_VARIABLE:
		// 	{
		// 	}
		// 	break;
		// 	case tenncor::node_proto_node_t::node_proto_node_t_PLACEHOLDER:
		// 	{
		// 	}
		// 	break;
		// 	case tenncor::node_proto_node_t::node_proto_node_t_CONSTANT:
		// 	{
		// 	}
		// 	break;
		// 	case tenncor::node_proto_node_t::node_proto_node_t_FUNCTOR:
		// 	{
		// 	}
		// 	break;
		// }
	}
}

std::string graph::register_node (inode* node)
{
	std::string uid = nnutils::uuid(node);
	auto it = order_.insert(order_.end(), uid);
	adjlist_[uid] = {node, it};
	return uid;
}

void graph::unregister_node (inode* node)
{
	auto it = adjlist_.find(node->get_uid());
	if (adjlist_.end() != it)
	{
		order_.erase(it->second.second);
		adjlist_.erase(it);
	}
}

}

#endif
