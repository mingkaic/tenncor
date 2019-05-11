///
/// save.hpp
/// pbm
///
/// Purpose:
/// Define functions for marshaling equation graph
///

#include <list>
#include <unordered_set>

#include "ade/traveler.hpp"
#include "ade/functor.hpp"

#include "pbm/data.hpp"

#ifndef PBM_SAVE_HPP
#define PBM_SAVE_HPP

namespace pbm
{

/// Map Tensptrs to a string path type
using PathedMapT = std::unordered_map<ade::TensptrT,StringsT>;

/// Graph serialization traveler
template <typename SAVER,
	typename std::enable_if<
		std::is_base_of<iSaver,SAVER>::value>::type* = nullptr>
struct GraphSaver final : public ade::iTraveler
{
	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		if (visited_.end() == visited_.find(leaf))
		{
			leaf->accept(stat);
			leaves_.push_back(leaf);
			visited_.emplace(leaf);
		}
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (visited_.end() == visited_.find(func))
		{
			func->accept(stat);
			funcs_.push_back(func);
			visited_.emplace(func);

			ade::ArgsT children = func->get_children();
			for (auto& child : children)
			{
				child.get_tensor()->accept(*this);
			}
		}
	}

	/// Marshal all equation graphs in roots vector to protobuf object
	void save (cortenn::Graph& out, PathedMapT labels = PathedMapT())
	{
		std::unordered_map<ade::iTensor*,StringsT> raw_labels;
		for (auto lpair : labels)
		{
			raw_labels[lpair.first.get()] = lpair.second;
		}

		// sort functions from the root with the smallest subtree to the largest
		// this ensures every children of a node appears before the parent,
		// as is the order of node creations
		funcs_.sort(
			[&](ade::iTensor* a, ade::iTensor* b)
			{
				return stat.graphsize_[a].upper_ < stat.graphsize_[b].upper_;
			});

		std::vector<ade::iFunctor*> funcs(funcs_.begin(), funcs_.end());
		std::vector<ade::iLeaf*> leaves(leaves_.begin(), leaves_.end());

		// all nodes in leaf appear before funcs
		std::unordered_map<ade::iTensor*,size_t> ordermap;
		size_t nleaves = leaves.size();
		for (size_t i = 0; i < nleaves; ++i)
		{
			ade::iLeaf* tens = leaves[i];
			ordermap[tens] = i;

			cortenn::Node* pb_node = out.add_nodes();
			auto it = raw_labels.find(tens);
			if (raw_labels.end() != it)
			{
				google::protobuf::RepeatedPtrField<std::string> vec(
					it->second.begin(), it->second.end());
				pb_node->mutable_labels()->Swap(&vec);
			}
			save_data(*pb_node->mutable_source(), tens);
		}
		for (size_t i = 0, n = funcs.size(); i < n; ++i)
		{
			ade::iFunctor* f = funcs[i];
			ordermap[f] = nleaves + i;

			cortenn::Node* pb_node = out.add_nodes();
			auto it = raw_labels.find(f);
			if (raw_labels.end() != it)
			{
				google::protobuf::RepeatedPtrField<std::string> vec(
					it->second.begin(), it->second.end());
				pb_node->mutable_labels()->Swap(&vec);
			}
			cortenn::Functor* func = pb_node->mutable_functor();
			ade::Opcode opcode = f->get_opcode();
			func->set_opname(opcode.name_);
			func->set_opcode(opcode.code_);
			const ade::ArgsT& children = f->get_children();
			for (auto& child : children)
			{
				cortenn::NodeArg* arg = func->add_args();
				ade::iTensor* tens = child.get_tensor().get();
				arg->set_idx(ordermap[tens]);
				std::vector<double> shaper =
					saver_.save_shaper(child.get_shaper());
				std::vector<double> coorder =
					saver_.save_coorder(child.get_coorder());
				google::protobuf::RepeatedField<double> shaper_vec(
					shaper.begin(), shaper.end());
				google::protobuf::RepeatedField<double> coorder_vec(
					coorder.begin(), coorder.end());
				arg->mutable_shaper()->Swap(&shaper_vec);
				arg->mutable_coord()->Swap(&coorder_vec);
				arg->set_fwd(child.map_io());
			}
		}
	}

	/// List of leaves visited (left to right)
	std::list<ade::iLeaf*> leaves_;

	/// List of functions visited (by depth-first)
	std::list<ade::iFunctor*> funcs_;

	/// Visited nodes
	std::unordered_set<ade::iTensor*> visited_;

	/// Internal traveler
	ade::GraphStat stat;

private:
	void save_data (cortenn::Source& out, ade::iLeaf* in)
	{
		const ade::Shape& shape = in->shape();
		size_t tcode = in->type_code();
		bool is_const = false;
		out.set_shape(std::string(shape.begin(), shape.end()));
		out.set_data(saver_.save_leaf(is_const, in));
		out.set_typecode(tcode);
		out.set_is_const(is_const);
	}

	SAVER saver_;
};

}

#endif // PBM_SAVE_HPP
