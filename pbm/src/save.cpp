#include "pbm/save.hpp"

#ifdef PBM_SAVE_HPP

namespace pbm
{

struct OrderedVisitor final : public teq::OnceTraveler
{
	void visit_leaf (teq::iLeaf* leaf) override
	{
		ordered_.push_back(leaf);
	}

	void visit_func (teq::iFunctor* func) override
	{
		auto children = func->get_children();
		for (const teq::iEdge& child : children)
		{
			child.get_tensor()->accept(*this);
		}
		ordered_.push_back(func);
	}

	std::vector<teq::iTensor*> ordered_;
};

static void tag_node (tenncor::Node* node,
	teq::iTensor* tens, tag::TagRegistry& registry)
{
	google::protobuf::Map<std::string,tenncor::Tag>* tags =
		node->mutable_tags();
	tag::TagRepsT reps = registry.get_tags(tens);
	for (auto reppair : reps)
	{
		google::protobuf::RepeatedPtrField<std::string> labels(
			reppair.second.begin(), reppair.second.end());
		google::protobuf::MapPair<std::string,tenncor::Tag> tagpair(
			reppair.first);
		tagpair.second.mutable_labels()->Swap(&labels);
		tags->insert(tagpair);
	}
}

void save_graph (tenncor::Graph& out, teq::TensptrsT roots,
	tag::TagRegistry& registry, LeafMarshF marshal_leaf)
{
	teq::GraphStat stat;
	OrderedVisitor orderer;
	for (teq::TensptrT root : roots)
	{
		if (nullptr != root)
		{
			root->accept(stat);
			root->accept(orderer);
		}
	}
	std::vector<teq::iLeaf*> leaves;
	std::vector<teq::iFunctor*> funcs;
	for (auto tens : orderer.ordered_)
	{
		if (stat.graphsize_[tens].upper_ == 0)
		{
			leaves.push_back(static_cast<teq::iLeaf*>(tens));
		}
		else
		{
			funcs.push_back(static_cast<teq::iFunctor*>(tens));
		}
	}
	std::sort(funcs.begin(), funcs.end(),
		[&](teq::iFunctor* a, teq::iFunctor* b)
		{
			return stat.graphsize_[a].upper_ < stat.graphsize_[b].upper_;
		});

	// map tens to index in leaves + funcs array
	std::unordered_map<teq::iTensor*,size_t> ordermap;
	size_t nleaves = leaves.size();
	for (size_t i = 0; i < nleaves; ++i)
	{
		teq::iLeaf* leaf = leaves[i];
		ordermap[leaf] = i;

		tenncor::Node* pb_node = out.add_nodes();
		pb_node->set_label(leaf->to_string());
		tag_node(pb_node, leaf, registry);

		tenncor::Source* pb_leaf = pb_node->mutable_source();

		const teq::Shape& shape = leaf->shape();
		google::protobuf::RepeatedField<google::protobuf::uint64> slist(
			shape.begin(), shape.end());
		pb_leaf->mutable_shape()->Swap(&slist);
		pb_leaf->set_data(marshal_leaf(leaf));
		pb_leaf->set_typelabel(leaf->type_label());
		pb_leaf->set_is_const(leaf->is_const());
	}
	for (size_t i = 0, n = funcs.size(); i < n; ++i)
	{
		teq::iFunctor* func = funcs[i];
		ordermap[func] = nleaves + i;

		tenncor::Node* pb_node = out.add_nodes();
		pb_node->set_label(func->to_string());
		tag_node(pb_node, func, registry);

		tenncor::Functor* pb_func = pb_node->mutable_functor();
		teq::Opcode opcode = func->get_opcode();
		pb_func->set_opname(opcode.name_);
		auto children = func->get_children();
		for (const teq::iEdge& child : children)
		{
			tenncor::NodeArg* pb_arg = pb_func->add_args();

			// serialize edge index
			pb_arg->set_idx(ordermap[child.get_tensor().get()]);

			// serialize edge attributes
			marsh::Maps mvalues;
			child.get_attrs(mvalues);

			auto pb_attrs = pb_arg->mutable_attrs();
			marshal_attrs(*pb_attrs, mvalues);
		}
	}
}

}

#endif
