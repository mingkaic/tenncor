#include "onnx/save.hpp"

#ifdef ONNX_SAVE_HPP

namespace onnx
{

// FuncPosT is the position of a functor tensor identified by height,
// breadth-wise of visit (in which left children have lower index than
// right children which have lower index than parent)
using FuncPosT = std::pair<size_t,size_t>;

struct OnnxMarshaler final : public teq::OnceTraveler
{
	OnnxMarshaler (GraphProto& graph, LeafMarshF leaf_marsh) :
		pb_graph_(graph), leaf_marsh_(leaf_marsh) {}

	std::unordered_set<const teq::iTensor*> roots_;

	std::unordered_map<const teq::iTensor*,size_t> tens_;

private:
	void visit_leaf (teq::iLeaf& leaf) override
	{
		std::string id = fmts::to_string(tens_.size());

		TensorAnnotation* pb_annotation =
			pb_graph_.add_quantization_annotation();
		pb_annotation->set_tensor_name(id);
		marshal_annotation(*pb_annotation, leaf);

		TensorProto* pb_tens = pb_graph_.add_initializer();
		pb_tens->set_name(id);
		teq::Shape shape = leaf.shape();
		google::protobuf::RepeatedField<int64_t> slist(
			shape.begin(), shape.end());
		pb_tens->mutable_dims()->Swap(&slist);
		leaf_marsh_(*pb_tens, leaf);

		roots_.emplace(&leaf);
		tens_.emplace(&leaf, tens_.size());
	}

	void visit_func (teq::iFunctor& func) override
	{
		auto children = func.get_children();
		for (teq::TensptrT ctens : children)
		{
			ctens->accept(*this);
		}

		if (auto lay = dynamic_cast<teq::iLayer*>(&func))
		{
			// todo: implement
		}

		std::string id = fmts::to_string(tens_.size());

		NodeProto* pb_node = pb_graph_.add_node();
		pb_node->set_name(id);
		pb_node->add_output(id);
		pb_node->set_op_type(func.to_string());
		for (teq::TensptrT ctens : children)
		{
			size_t subid = estd::must_getf(tens_, ctens.get(),
				"cannot find child traversed %s",
				ctens->to_string().c_str());
			pb_node->add_input(fmts::to_string(subid));
			roots_.erase(ctens.get());
		}
		auto pb_attrs = pb_node->mutable_attribute();
		marshal_attrs(*pb_attrs, func);
		roots_.emplace(&func);
		tens_.emplace(&func, tens_.size());
	}

	GraphProto& pb_graph_;

	LeafMarshF leaf_marsh_;
};

void save_graph (GraphProto& pb_graph,
	teq::TensptrsT roots, LeafMarshF marshal_leaf)
{
	OnnxMarshaler marshal(pb_graph, marshal_leaf);
	for (teq::TensptrT root : roots)
	{
		if (nullptr != root)
		{
			root->accept(marshal);
		}
	}

	std::vector<const teq::iTensor*> rtens(marshal.roots_.begin(), marshal.roots_.end());
	std::sort(rtens.begin(), rtens.end(),
		[&marshal](const teq::iTensor* a, const teq::iTensor* b)
		{
			return marshal.tens_.at(a) < marshal.tens_.at(b);
		});
	for (const teq::iTensor* root : rtens)
	{
		ValueInfoProto* pb_output = pb_graph.add_output();
		pb_output->set_name(fmts::to_string(marshal.tens_.at(root)));
		marshal_io(*pb_output, root->shape());
	}
}

}

#endif
