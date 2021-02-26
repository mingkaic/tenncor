#include "internal/onnx/load.hpp"

#ifdef ONNX_LOAD_HPP

namespace onnx
{

teq::TensptrsT load_graph (TensptrIdT& identified_tens,
	const GraphProto& pb_graph, const iUnmarshFuncs& unmarshaler)
{
	const auto& pb_annotations = pb_graph.quantization_annotation();
	auto annotations = unmarshal_annotation(pb_annotations);

	const auto& pb_inputs = pb_graph.input();
	for (const ValueInfoProto& pb_input : pb_inputs)
	{
		std::string id = pb_input.name();
		if (estd::has(identified_tens.right, id))
		{
			continue; // allow previously defined ids
		}
		std::string name;
		if (estd::has(annotations, id))
		{
			AnnotationsT& ans = annotations[id];
			name = estd::try_get(ans, leafname_key, "");
		}

		const TypeProto::Tensor& tens_type = pb_input.type().tensor_type();
		const auto& dims = tens_type.shape().dim();
		TensorProto pb_ten;
		for (const auto& dim : dims)
		{
			pb_ten.add_dims(dim.dim_value());
		}
		pb_ten.set_data_type(tens_type.elem_type());
		identified_tens.insert({unmarshaler.unmarsh_leaf(
			pb_ten, teq::PLACEHOLDER, name), id});
	}

	const auto& pb_tens = pb_graph.initializer();
	for (const TensorProto& pb_ten : pb_tens)
	{
		std::string id = pb_ten.name();
		if (estd::has(identified_tens.right, id))
		{
			continue; // allow previously defined ids
		}
		std::string name;
		teq::Usage usage = teq::UNKNOWN_USAGE;
		if (estd::has(annotations, id))
		{
			AnnotationsT& ans = annotations[id];
			name = estd::try_get(ans, leafname_key, "");
			usage = teq::get_named_usage(
				estd::try_get(ans, leafusage_key, ""));
		}
		identified_tens.insert({unmarshaler.unmarsh_leaf(
			pb_ten, usage, name), id});
	}

	const auto& pb_stens = pb_graph.sparse_initializer();
	for (const SparseTensorProto& pb_sten : pb_stens)
	{
		const TensorProto& pb_ten = pb_sten.values();
		std::string id = pb_ten.name();
		if (estd::has(identified_tens.right, id))
		{
			continue; // allow previously defined ids
		}
		std::string name;
		teq::Usage usage = teq::UNKNOWN_USAGE;
		if (estd::has(annotations, id))
		{
			AnnotationsT& ans = annotations[id];
			name = estd::try_get(ans, leafname_key, "");
			usage = teq::get_named_usage(
				estd::try_get(ans, leafusage_key, ""));
		}
		identified_tens.insert({unmarshaler.unmarsh_leaf(
			pb_sten, usage, name), id});
	}

	const auto& pb_nodes = pb_graph.node();
	for (const NodeProto& pb_node : pb_nodes)
	{
		std::string opname = pb_node.op_type();
		assert(opname.size() > 0);
		marsh::Maps attrs;
		const auto& pb_attrs = pb_node.attribute();
		const auto& inputs = pb_node.input();
		teq::TensptrsT args;
		args.reserve(inputs.size());
		std::transform(inputs.begin(), inputs.end(),
			std::back_inserter(args),
			[&identified_tens](std::string input)
			{
				return estd::must_getf(identified_tens.right, input,
					"failed to find input %s", input.c_str());
			});
		teq::TensptrT tens;
		if (const GraphProto* sub = unmarshal_attrs(
			attrs, pb_attrs, identified_tens))
		{
			teq::TensptrsT roots = load_graph(
				identified_tens, *sub, unmarshaler);
			tens = unmarshaler.unmarsh_layr(
				opname, roots.front(), args.front(), std::move(attrs));
		}
		else
		{
			std::string id = pb_node.name();
			if (estd::has(identified_tens.right, id))
			{
				global::fatalf("duplicate id %s", id.c_str());
			}
			tens = unmarshaler.unmarsh_func(opname, args, std::move(attrs));
			identified_tens.insert({tens, id});
		}
	}

	teq::TensptrsT roots;
	const auto& pb_outputs = pb_graph.output();
	for (const ValueInfoProto& pb_output : pb_outputs)
	{
		roots.push_back(identified_tens.right.at(pb_output.name()));
	}
	return roots;
}

}

#endif
