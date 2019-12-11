#include <algorithm>

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

void save_graph (GraphProto& pb_graph,
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

	size_t i = 0;
	std::unordered_set<const teq::iTensor*> root_tens;
	std::unordered_map<const teq::iTensor*,size_t> tens;
	for (const teq::iLeaf* leaf : leaves)
	{
		std::string id = fmts::to_string(i);

		TensorAnnotation* pb_annotation =
			pb_graph.add_quantization_annotation();
		pb_annotation->set_tensor_name(id);
		marshal_annotation(*pb_annotation, *leaf);

		TensorProto* pb_tens = pb_graph.add_initializer();
		pb_tens->set_name(id);
		teq::Shape shape = leaf->shape();
		google::protobuf::RepeatedField<int64_t> slist(
			shape.begin(), shape.end());
		pb_tens->mutable_dims()->Swap(&slist);
		marshal_leaf(*pb_tens, *leaf);

		// // todo: distinguish weight and placeholders
		// ValueInfoProto* pb_input = pb_graph.add_input();
		// pb_input->set_name(id);
		// marshal_io(*pb_input, pb_tens->data_type(), leaf->shape());

		root_tens.emplace(leaf);
		tens.emplace(leaf, i);
		++i;
	}
	for (const teq::iFunctor* func : funcs)
	{
		std::string id = fmts::to_string(i);

		NodeProto* pb_node = pb_graph.add_node();
		pb_node->set_name(id);
		pb_node->add_output(id);
		pb_node->set_op_type(func->get_opcode().name_);
		auto children = func->get_children();
		for (const teq::iEdge& child : children)
		{
			auto ctens = child.get_tensor().get();
			pb_node->add_input(fmts::to_string(tens[ctens]));
			root_tens.erase(ctens);
		}
		auto pb_attrs = pb_node->mutable_attribute();
		marshal_attrs(*pb_attrs, func);

		root_tens.emplace(func);
		tens.emplace(func, i);
		++i;
	}

	std::vector<const teq::iTensor*> rtens(root_tens.begin(), root_tens.end());
	std::sort(rtens.begin(), rtens.end(),
		[&tens](const teq::iTensor* a, const teq::iTensor* b)
		{
			return tens.at(a) < tens.at(b);
		});
	for (const teq::iTensor* root : rtens)
	{
		ValueInfoProto* pb_output = pb_graph.add_output();
		pb_output->set_name(fmts::to_string(tens.at(root)));
		marshal_io(*pb_output, 0, root->shape());
	}
}

void save_graph (GraphProto& pb_graph, teq::EdgeptrsT roots, LeafMarshF marshal_leaf)
{
	//
}

}

#endif
