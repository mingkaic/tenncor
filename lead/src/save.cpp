//
//  save.cpp
//  lead
//

#include "lead/save.hpp"
#include "lead/include/packer.hpp"

#include "kiln/variable.hpp"
#include "kiln/placeholder.hpp"
#include "kiln/constant.hpp"
#include "kiln/functor.hpp"

#ifdef LEAD_SAVE_HPP

namespace lead
{

void save_tensor (tenncor::TensorPb& out, clay::State state)
{
	if (state.data_.expired())
	{
		throw std::exception(); // todo: add context
	}
	clay::Shape& shape = state.shape_;
	clay::DTYPE type = state.dtype_;
	// set TensorPb data (1)
	pack_data(out.mutable_data(), state.data_.lock(), shape.n_elems(), type);

	// set TensorPb shape (2)
	std::vector<size_t> slist = shape.as_list();
	google::protobuf::RepeatedField<uint64_t> field(slist.begin(), slist.end());
	out.mutable_shape()->Swap(&field);

	// set TensorPb type (3)
	out.set_type(type);
}

void save_data (tenncor::DataRepoPb& out, const kiln::Graph& graph)
{
	// set GraphPb gid (1)
	out.set_gid(graph.get_gid());
	out.clear_data_map();
	// set GraphPb node_map (2)
	google::protobuf::Map<std::string,tenncor::TensorPb>&
		tmap = *(out.mutable_data_map());
	for (const kiln::Identifier* id : graph)
	{
		if (dynamic_cast<const kiln::Variable*>(id) ||
			dynamic_cast<const kiln::Placeholder*>(id))
		{
			clay::State state = id->get_state();
			tenncor::TensorPb& tenout = tmap[id->get_uid()];
			save_tensor(tenout, state);
		}
	}
}

void save_graph (tenncor::GraphPb& out, const kiln::Graph& graph)
{
	// set GraphPb gid (1)
	out.set_gid(graph.get_gid());
	out.clear_node_map();
	out.clear_create_order();
	// set GraphPb node_map (3)
	google::protobuf::Map<std::string,tenncor::NodePb>&
		nmap = *(out.mutable_node_map());
	for (const kiln::Identifier* id : graph)
	{
		std::string uid = id->get_uid();
		tenncor::NodePb& node_dest = nmap[uid];

		// set NodePb label (2)
		node_dest.set_label(id->get_label());

		if (const kiln::Functor* f = dynamic_cast<const kiln::Functor*>(id))
		{
			// set NodePb type (1)
			node_dest.set_type(tenncor::NodePb::FUNCTOR);

			tenncor::FunctorPb fpb;
			fpb.set_opcode(f->opcode_);
			auto args = f->get_args();
			for (std::string& arg_id : args)
			{
				fpb.add_args(arg_id);
			}

			// set NodePb detail (3)
			node_dest.mutable_detail()->PackFrom(fpb);
		}
		else
		{
			tenncor::NodePb::NodeT type;
			if (dynamic_cast<const kiln::Constant*>(id))
			{
				type = tenncor::NodePb::CONSTANT;

				// set NodePb detail (3)
				tenncor::TensorPb tenout;
				clay::State state = id->get_state();
				save_tensor(tenout, state);
				node_dest.mutable_detail()->PackFrom(tenout);
			}
			else if (dynamic_cast<const kiln::Variable*>(id))
			{
				type = tenncor::NodePb::VARIABLE;
			}
			else if (dynamic_cast<const kiln::Placeholder*>(id))
			{
				type = tenncor::NodePb::PLACEHOLDER;
			}
			else
			{
				throw std::exception(); // todo: add context
			}

			// set NodePb type (1)
			node_dest.set_type(type);
		}

		// set GraphPb create_order (3)
		out.add_create_order(uid);
	}
}

}

#endif
