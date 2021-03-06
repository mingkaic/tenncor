///
/// save.hpp
/// onnx
///
/// Purpose:
/// Define functions for saving teq graph
///

#ifndef ONNX_SAVE_HPP
#define ONNX_SAVE_HPP

#include "internal/onnx/marshal.hpp"

namespace onnx
{

struct iMarshFuncs
{
	virtual ~iMarshFuncs (void) = default;

	virtual size_t get_typecode (const teq::iTensor& tens) const = 0;

	virtual void marsh_leaf (
		TensorProto& pb_tens, const teq::iLeaf& leaf) const = 0;
};

struct OnnxMarshaler final : public teq::iTraveler
{
	OnnxMarshaler (GraphProto& graph, const TensIdT& identified,
		const iMarshFuncs& marshaler, teq::TensSetT stops = {}) :
		pb_graph_(graph), identified_(identified),
		marshaler_(marshaler), stops_(stops)
	{
		for (auto& node : graph.node())
		{
			preexisting_ids_.emplace(node.name());
		}
		for (auto& init : graph.initializer())
		{
			preexisting_ids_.emplace(init.name());
		}
		for (auto& input : graph.input())
		{
			preexisting_ids_.emplace(input.name());
		}
	}

	void visit (teq::iLeaf& leaf) override
	{
		if (estd::has(tens_, &leaf))
		{
			return;
		}
		std::string id = get_id(leaf);
		roots_.emplace(&leaf);
		tens_.emplace(&leaf, id);
		teq::Shape shape = leaf.shape();
		if (estd::has(stops_, &leaf))
		{
			if (false == estd::has(preexisting_ids_, id))
			{
				ValueInfoProto* pb_place = pb_graph_.add_input();
				pb_place->set_name(id);

				TypeProto* pb_type = pb_place->mutable_type();
				TypeProto::Tensor* tens_type = pb_type->mutable_tensor_type();
				tens_type->set_elem_type(marshaler_.get_typecode(leaf));
				auto dims = tens_type->mutable_shape()->mutable_dim();
				for (teq::DimT d : shape)
				{
					dims->Add()->set_dim_value(d);
				}
			}
			return;
		}

		auto usage = leaf.get_usage();

		TensorAnnotation* pb_annotation =
			pb_graph_.add_quantization_annotation();
		pb_annotation->set_tensor_name(id);
		marshal_annotation(*pb_annotation, leaf);

		if (teq::PLACEHOLDER == usage)
		{
			ValueInfoProto* pb_place = pb_graph_.add_input();
			pb_place->set_name(id);

			TypeProto* pb_type = pb_place->mutable_type();
			TypeProto::Tensor* tens_type = pb_type->mutable_tensor_type();
			tens_type->set_elem_type(marshaler_.get_typecode(leaf));
			auto dims = tens_type->mutable_shape()->mutable_dim();
			for (teq::DimT d : shape)
			{
				dims->Add()->set_dim_value(d);
			}
		}
		else // IMMUTABLE or Variable
		{
			TensorProto* pb_tens = pb_graph_.add_initializer();
			pb_tens->set_name(id);
			google::protobuf::RepeatedField<int64_t> slist(
				shape.begin(), shape.end());
			pb_tens->mutable_dims()->Swap(&slist);
			marshaler_.marsh_leaf(*pb_tens, leaf);
		}
	}

	void visit (teq::iFunctor& func) override
	{
		if (estd::has(tens_, &func))
		{
			return;
		}

		if (estd::has(stops_, &func))
		{
			std::string id = get_id(func);
			roots_.emplace(&func);
			tens_.emplace(&func, id);

			if (false == estd::has(preexisting_ids_, id))
			{
				ValueInfoProto* pb_place = pb_graph_.add_input();
				pb_place->set_name(id);

				TypeProto* pb_type = pb_place->mutable_type();
				TypeProto::Tensor* tens_type = pb_type->mutable_tensor_type();
				tens_type->set_elem_type(marshaler_.get_typecode(func));
				auto dims = tens_type->mutable_shape()->mutable_dim();
				auto shape = func.shape();
				for (teq::DimT d : shape)
				{
					dims->Add()->set_dim_value(d);
				}
			}

			return;
		}

		if (auto lattr = func.get_attr(teq::layer_attr))
		{
			marshal_layer(func,
				static_cast<const teq::LayerObj*>(lattr));
		}
		else
		{
			marshal_func(func);
		}
	}

	std::unordered_set<const teq::iTensor*> roots_;

	teq::CTensMapT<std::string> tens_;

private:
	void marshal_func (teq::iFunctor& func)
	{
		roots_.emplace(&func);
		auto deps = func.get_args();
		if (func.size() > 0)
		{
			marsh::Maps attrs;
			marsh::get_attrs(attrs, func);

			teq::FindTensAttr finder;
			attrs.accept(finder);
			deps.insert(deps.end(),
				finder.tens_.begin(), finder.tens_.end());
		}
		teq::multi_visit(*this, deps);
		auto attrs = func.ls_attrs();
		for (auto attr : attrs)
		{
			if (auto tensattr = dynamic_cast<
				const teq::TensorObj*>(func.get_attr(attr)))
			{
				auto tens = tensattr->get_tensor();
				assert(nullptr != tens);
				tens->accept(*this);
				roots_.erase(tens.get());
			}
		}

		std::string id = get_id(func);
		NodeProto* pb_node = pb_graph_.add_node();
		pb_node->set_name(id);
		pb_node->add_output(id);
		pb_node->set_op_type(func.to_string());

		auto children = func.get_args();
		for (teq::TensptrT ctens : children)
		{
			pb_node->add_input(estd::must_getf(tens_, ctens.get(),
				"cannot find child traversed %s", ctens->to_string().c_str()));
			roots_.erase(ctens.get());
		}
		auto pb_attrs = pb_node->mutable_attribute();
		marshal_attrs(*pb_attrs, func, tens_);

		tens_.emplace(&func, id);
	}

	void marshal_layer (teq::iFunctor& func, const teq::LayerObj* layer)
	{
		// for layers, skip the subgraph and marshal inputs first
		teq::TensptrT input = layer->get_tensor();
		assert(nullptr != input);
		input->accept(*this);
		roots_.erase(input.get());

		NodeProto* pb_node = pb_graph_.add_node();
		pb_node->set_op_type(layer->get_opname());
		auto pb_attrs = pb_node->mutable_attribute();
		AttributeProto* inner_workings = pb_attrs->Add();
		inner_workings->set_name(teq::layer_attr);
		inner_workings->set_type(AttributeProto::GRAPH);
		GraphProto* subgraph = inner_workings->mutable_g();

		std::string subid = estd::must_getf(tens_, input.get(),
			"cannot find child traversed %s",
			input->to_string().c_str());
		pb_node->add_input(subid);

		auto sub_input = subgraph->add_input();
		sub_input->set_name(subid);

		TypeProto* pb_type = sub_input->mutable_type();
		TypeProto::Tensor* tens_type = pb_type->mutable_tensor_type();
		tens_type->set_elem_type(marshaler_.get_typecode(*input));
		auto dims = tens_type->mutable_shape()->mutable_dim();
		auto cshape = input->shape();
		for (teq::DimT d : cshape)
		{
			dims->Add()->set_dim_value(d);
		}
		roots_.erase(input.get());

		auto substops = stops_;
		substops.emplace(input.get());
		OnnxMarshaler submarsh(*subgraph, identified_, marshaler_, substops);
		submarsh.roots_ = roots_;
		submarsh.tens_ = tens_;
		submarsh.marshal_func(func);
		roots_ = submarsh.roots_;
		tens_ = submarsh.tens_;

		std::string id = tens_.at(&func);
		ValueInfoProto* pb_output = subgraph->add_output();

		pb_output->set_name(id);
		pb_node->set_name(id);
		pb_node->add_output(id);

		marshal_io(*pb_output, func.shape());
		marshal_attrs(*pb_attrs, func, tens_);
	}

	std::string get_id (teq::iTensor& tens) const
	{
		if (estd::has(identified_.left, &tens))
		{
			return identified_.left.at(&tens);
		}
		auto gen = global::get_generator();
		auto out = gen->get_str();
		while (estd::has(preexisting_ids_, out))
		{
			out = gen->get_str();
		}
		return out;
	}

	GraphProto& pb_graph_;

	const TensIdT& identified_;

	const iMarshFuncs& marshaler_;

	teq::TensSetT stops_;

	types::StrUSetT preexisting_ids_;
};

template <typename TS> // todo: use concept tensptr_range
void save_graph (GraphProto& pb_graph, const TS& roots,
	const iMarshFuncs& marshaler, const TensIdT& identified = {},
	const teq::TensSetT& stops = {})
{
	OnnxMarshaler marshal(pb_graph, identified, marshaler, stops);
	teq::multi_visit(marshal, roots);

	std::vector<const teq::iTensor*> rtens(
		marshal.roots_.begin(), marshal.roots_.end());
#ifdef ORDERED_SAVE
	std::sort(rtens.begin(), rtens.end(),
    [&marshal](const teq::iTensor* a, const teq::iTensor* b)
    {
        return marshal.tens_.at(a) < marshal.tens_.at(b);
    });
#endif
	for (const teq::iTensor* root : rtens)
	{
		ValueInfoProto* pb_output = pb_graph.add_output();
		pb_output->set_name(marshal.tens_.at(root));
		marshal_io(*pb_output, root->shape());
	}
}

}

#endif // ONNX_SAVE_HPP
