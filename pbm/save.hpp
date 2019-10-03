///
/// save.hpp
/// pbm
///
/// Purpose:
/// Define functions for marshaling equation graph
///

#include <list>
#include <unordered_set>

#include "teq/traveler.hpp"
#include "teq/functor.hpp"

#include "pbm/data.hpp"

#ifndef PBM_SAVE_HPP
#define PBM_SAVE_HPP

namespace pbm
{

/// Map Tensptrs to a string path type
using PathedMapT = std::unordered_map<teq::TensptrT,StringsT>;

/// Graph serialization traveler
template <typename SAVER,
	typename std::enable_if<
		std::is_base_of<iSaver,SAVER>::value>::type* = nullptr>
struct GraphSaver final : public teq::iTraveler
{
	GraphSaver (tag::TagRegistry& registry = tag::get_reg()) :
		registry_(registry) {}

	/// Implementation of iTraveler
	void visit (teq::iLeaf* leaf) override
	{
		if (false == estd::has(visited_, leaf))
		{
			leaf->accept(stat);
			leaves_.push_back(leaf);
			visited_.emplace(leaf);
		}
	}

	/// Implementation of iTraveler
	void visit (teq::iFunctor* func) override
	{
		if (false == estd::has(visited_, func))
		{
			func->accept(stat);
			funcs_.push_back(func);
			visited_.emplace(func);

			teq::ArgsT children = func->get_children();
			for (auto& child : children)
			{
				child.get_tensor()->accept(*this);
			}
		}
	}

	/// Marshal all equation graphs in roots vector to protobuf object
	void save (cortenn::Graph& out, PathedMapT labels = PathedMapT())
	{
		// sort functions from the root with the smallest subtree to the largest
		// this ensures every children of a node appears before the parent,
		// as is the order of node creations
		funcs_.sort(
			[&](teq::iTensor* a, teq::iTensor* b)
			{
				return stat.graphsize_[a].upper_ < stat.graphsize_[b].upper_;
			});

		std::vector<teq::iFunctor*> funcs(funcs_.begin(), funcs_.end());
		std::vector<teq::iLeaf*> leaves(leaves_.begin(), leaves_.end());

		// all nodes in leaf appear before funcs
		std::unordered_map<teq::iTensor*,size_t> ordermap;
		size_t nleaves = leaves.size();
		for (size_t i = 0; i < nleaves; ++i)
		{
			teq::iLeaf* tens = leaves[i];
			ordermap[tens] = i;

			cortenn::Node* pb_node = out.add_nodes();
			pb_node->set_label(tens->to_string());
			tag_node(pb_node, tens, registry_);
			save_data(*pb_node->mutable_source(), tens);
		}
		for (size_t i = 0, n = funcs.size(); i < n; ++i)
		{
			teq::iFunctor* f = funcs[i];
			ordermap[f] = nleaves + i;

			cortenn::Node* pb_node = out.add_nodes();
			pb_node->set_label(f->to_string());
			tag_node(pb_node, f, registry_);
			cortenn::Functor* func = pb_node->mutable_functor();
			teq::Opcode opcode = f->get_opcode();
			func->set_opname(opcode.name_);
			const teq::ArgsT& children = f->get_children();
			for (auto& child : children)
			{
				cortenn::NodeArg* arg = func->add_args();
				teq::iTensor* tens = child.get_tensor().get();
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
	std::list<teq::iLeaf*> leaves_;

	/// List of functions visited (by depth-first)
	std::list<teq::iFunctor*> funcs_;

	/// Visited nodes
	teq::TensSetT visited_;

	/// Internal traveler
	teq::GraphStat stat;

private:
	void save_data (cortenn::Source& out, teq::iLeaf* in)
	{
		const teq::Shape& shape = in->shape();
		google::protobuf::RepeatedField<google::protobuf::uint64> slist(
			shape.begin(), shape.end());
		out.mutable_shape()->Swap(&slist);
		out.set_data(saver_.save_leaf(in));
		out.set_typelabel(in->type_label());
		out.set_is_const(in->is_const());
	}

	void tag_node (cortenn::Node* node,
		teq::iTensor* tens, tag::TagRegistry& registry)
	{
		google::protobuf::Map<std::string,cortenn::Tag>* tags =
			node->mutable_tags();
		tag::TagRepsT reps = registry.get_tags(tens);
		for (auto reppair : reps)
		{
			google::protobuf::RepeatedPtrField<std::string> labels(
				reppair.second.begin(), reppair.second.end());
			google::protobuf::MapPair<std::string,cortenn::Tag> tagpair(
				reppair.first);
			tagpair.second.mutable_labels()->Swap(&labels);
			tags->insert(tagpair);
		}
	}

	SAVER saver_;

	tag::TagRegistry& registry_;
};

}

#endif // PBM_SAVE_HPP
