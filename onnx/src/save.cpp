#include "onnx/save.hpp"

#ifdef ONNX_SAVE_HPP

namespace onnx
{

struct OnnxMarshaler final : public teq::OnceTraveler
{
	OnnxMarshaler (GraphProto& graph, LeafMarshF leaf_marsh,
		teq::TensSetT stops = {}, std::string id_prefix = "") :
		pb_graph_(graph), leaf_marsh_(leaf_marsh),
		id_prefix_(id_prefix), stops_(stops) {}

	std::unordered_set<const teq::iTensor*> roots_;

	std::unordered_map<const teq::iTensor*,size_t> tens_;

private:
	void visit_leaf (teq::iLeaf& leaf) override
	{
		if (estd::has(stops_, &leaf))
		{
			return;
		}
		std::string id = fmts::to_string(tens_.size());
		auto usage = leaf.get_usage();

		TensorAnnotation* pb_annotation =
			pb_graph_.add_quantization_annotation();
		pb_annotation->set_tensor_name(id);
		marshal_annotation(*pb_annotation, leaf);
		teq::Shape shape = leaf.shape();

		if (teq::Placeholder == usage)
		{
			ValueInfoProto* pb_place = pb_graph_.add_input();
			pb_place->set_name(id);

			TypeProto* pb_type = pb_place->mutable_type();
			TypeProto::Tensor* tens_type = pb_type->mutable_tensor_type();
			auto dims = tens_type->mutable_shape()->mutable_dim();
			for (teq::DimT d : shape)
			{
				dims->Add()->set_dim_value(d);
			}
		}
		else // Immutable or Variable
		{
			TensorProto* pb_tens = pb_graph_.add_initializer();
			pb_tens->set_name(id);
			google::protobuf::RepeatedField<int64_t> slist(
				shape.begin(), shape.end());
			pb_tens->mutable_dims()->Swap(&slist);
			leaf_marsh_(*pb_tens, leaf);
		}
		roots_.emplace(&leaf);
		tens_.emplace(&leaf, tens_.size());
	}

	void visit_func (teq::iFunctor& func) override
	{
		if (estd::has(stops_, &func))
		{
			return;
		}
		auto children = func.get_children();
		for (teq::TensptrT ctens : children)
		{
			ctens->accept(*this);
		}

		std::string id = fmts::to_string(tens_.size());
		NodeProto* pb_node = pb_graph_.add_node();
		pb_node->set_name(id);
		pb_node->add_output(id);
		pb_node->set_op_type(func.to_string());

		auto pb_attrs = pb_node->mutable_attribute();
		if (auto lay = dynamic_cast<teq::iLayer*>(&func))
		{
			// todo: implement
			AttributeProto* inner_workings = pb_attrs->Add();
			inner_workings->set_name(subgraph_key);
			inner_workings->set_type(AttributeProto::GRAPH);
			GraphProto* subgraph = inner_workings->mutable_g();
			for (teq::TensptrT ctens : children)
			{
				size_t subid = estd::must_getf(tens_, ctens.get(),
					"cannot find child traversed %s",
					ctens->to_string().c_str());
				pb_node->add_input(fmts::to_string(subid));

				auto sub_input = subgraph->add_input();
				sub_input->set_name(fmts::to_string(subid));

				TypeProto* pb_type = sub_input->mutable_type();
				TypeProto::Tensor* tens_type = pb_type->mutable_tensor_type();
				auto dims = tens_type->mutable_shape()->mutable_dim();
				auto cshape = ctens->shape();
				for (teq::DimT d : cshape)
				{
					dims->Add()->set_dim_value(d);
				}
				roots_.erase(ctens.get());
			}
			teq::TensSetT childset;
			std::transform(children.begin(), children.end(),
				std::inserter(childset, childset.end()),
				[](teq::TensptrT child) { return child.get(); });
			OnnxMarshaler submarsh(*subgraph, leaf_marsh_,
				childset, id_prefix_ + "::" + id);
			auto subroot = lay->get_root();
			subroot->accept(submarsh);
		}
		else
		{
			for (teq::TensptrT ctens : children)
			{
				size_t subid = estd::must_getf(tens_, ctens.get(),
					"cannot find child traversed %s",
					ctens->to_string().c_str());
				pb_node->add_input(fmts::to_string(subid));
				roots_.erase(ctens.get());
			}
		}
		marshal_attrs(*pb_attrs, func);
		roots_.emplace(&func);
		tens_.emplace(&func, tens_.size());
	}

	GraphProto& pb_graph_;

	LeafMarshF leaf_marsh_;

	std::string id_prefix_;

	teq::TensSetT stops_;
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
