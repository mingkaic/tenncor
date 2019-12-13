#include <algorithm>

#include "onnx/save.hpp"

#ifdef ONNX_SAVE_HPP

namespace onnx
{

// FuncPosT is the position of a functor tensor identified by height,
// breadth-wise of visit (in which left children have lower index than
// right children which have lower index than parent)
using FuncPosT = std::pair<size_t,size_t>;

struct OrderedLocator final : public teq::OnceTraveler
{
	void visit_leaf (teq::iLeaf& leaf) override
	{
		leaves_.push_back(&leaf);
	}

	void visit_func (teq::iFunctor& func) override
	{
		size_t height = 0;
		auto children = func.get_children();
		for (teq::TensptrT child : children)
		{
			child->accept(*this);
			if (auto f = dynamic_cast<teq::iFunctor*>(child.get()))
			{
				height = std::max(height, funcs_[f].first);
			}
		}
		funcs_.emplace(&func, FuncPosT{height, funcs_.size()});
	}

	void visit_place (teq::Placeholder& place) override
	{
		places_.push_back(&place);
	}

	std::vector<teq::iLeaf*> leaves_;

	std::unordered_map<teq::iFunctor*,FuncPosT> funcs_;

	std::vector<teq::Placeholder*> places_;
};

void save_graph (GraphProto& pb_graph,
	teq::TensptrsT roots, LeafMarshF marshal_leaf)
{
	OrderedLocator ord;
	for (teq::TensptrT root : roots)
	{
		if (nullptr != root)
		{
			root->accept(ord);
		}
	}

	std::vector<teq::Placeholder*>& places = ord.places_;
	std::vector<teq::iLeaf*>& leaves = ord.leaves_;
	std::vector<teq::iFunctor*> funcs;
	{
		funcs.reserve(ord.funcs_.size());
		for (auto fpair : ord.funcs_)
		{
			funcs.push_back(fpair.first);
		}
		std::sort(funcs.begin(), funcs.end(),
			[&](teq::iFunctor* a, teq::iFunctor* b)
			{
				auto& apos = ord.funcs_[a];
				auto& bpos = ord.funcs_[b];
				if (apos.first == bpos.first)
				{
					return apos.second < bpos.second;
				}
				return apos.first < bpos.first;
			});
	}

	size_t i = 0;
	std::unordered_set<const teq::iTensor*> root_tens;
	std::unordered_map<const teq::iTensor*,size_t> tens;
	for (const teq::Placeholder* place : places)
	{
		std::string id = fmts::to_string(i);

		ValueInfoProto* pb_input = pb_graph.add_input();
		pb_input->set_name(id);
		marshal_io(*pb_input, 0, place->shape_sign());
		root_tens.emplace(place);
		tens.emplace(place, i);
		++i;
	}
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
		for (teq::TensptrT ctens : children)
		{
			pb_node->add_input(fmts::to_string(tens[ctens.get()]));
			root_tens.erase(ctens.get());
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

}

#endif
