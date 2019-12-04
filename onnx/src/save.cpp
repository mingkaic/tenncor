#include <algorithm>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "onnx/save.hpp"

#ifdef ONNX_SAVE_HPP

namespace onnx
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

void save_graph (GraphProto& out,
	teq::TensptrsT roots, LeafMarshF marshal_leaf)
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
	{
		std::unordered_map<teq::iFunctor*,size_t> forder;
		for (auto tens : orderer.ordered_)
		{
			if (stat.graphsize_[tens].upper_ == 0)
			{
				leaves.push_back(static_cast<teq::iLeaf*>(tens));
			}
			else
			{
				auto f = static_cast<teq::iFunctor*>(tens);
				forder[f] = funcs.size();
				funcs.push_back(f);
			}
		}
		std::sort(funcs.begin(), funcs.end(),
			[&](teq::iFunctor* a, teq::iFunctor* b)
			{
				if (stat.graphsize_[a].upper_ == stat.graphsize_[b].upper_)
				{
					return forder[a] < forder[b];
				}
				return stat.graphsize_[a].upper_ < stat.graphsize_[b].upper_;
			});
	}

	boost::uuids::random_generator uuid_gen;
	std::unordered_map<const teq::iTensor*,std::string> tens;
	for (const teq::iLeaf* leaf : leaves)
	{
		TensorProto* pb_tens = out.add_initializer();
		marshal_leaf(pb_tens, leaf);

		NodeProto* pb_node = out.add_node();
		std::string id = boost::uuids::to_string(uuid_gen());
		pb_node->set_name(id);
		if (leaf->is_const())
		{
			AttributeProto* pb_attr = pb_node->add_attribute();
			pb_attr->set_type(AttributeProto::FLOATS);
			pb_attr->set_name(leafconst_key);
		}
		tens.emplace(leaf, id);
	}
	for (const teq::iFunctor* func : funcs)
	{
		teq::Shape shape = func->shape();
		TensorProto* pb_tens = out.add_initializer();
		google::protobuf::RepeatedField<int64_t> slist(
			shape.begin(), shape.end());
		pb_tens->mutable_dims()->Swap(&slist);

		NodeProto* pb_node = out.add_node();
		std::string id = boost::uuids::to_string(uuid_gen());
		pb_node->set_name(id);
		pb_node->set_op_type(func->get_opcode().name_);
		auto children = func->get_children();
		for (const teq::iEdge& child : children)
		{
			pb_node->add_input(tens[child.get_tensor().get()]);
		}
		auto pb_attrs = pb_node->mutable_attribute();
		marshal_attrs(*pb_attrs, func);
		tens.emplace(func, id);
	}
}

}

#endif
