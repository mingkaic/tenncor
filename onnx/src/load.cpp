#include "onnx/load.hpp"

#ifdef ONNX_LOAD_HPP

namespace onnx
{

void load_graph (teq::TensptrsT& roots,
	const GraphProto& pb_graph, iUnmarshFuncs& unmarshaler,
	std::unordered_map<std::string,teq::TensptrT> created_tens)
{
	const auto& pb_annotations = pb_graph.quantization_annotation();
	auto annotations = unmarshal_annotation(pb_annotations);

	const auto& pb_inputs = pb_graph.input();
	for (const ValueInfoProto& pb_input : pb_inputs)
	{
		std::string id = pb_input.name();
		std::string name;
		if (estd::has(annotations, id))
		{
			AnnotationsT& ans = annotations[id];
			name = estd::try_get(ans, leafname_key, "");
		}

		const TypeProto& type = pb_input.type();
		const TypeProto::Tensor& tens_type = type.tensor_type();
		const auto& dims = tens_type.shape().dim();
		TensorProto pb_ten;
		for (const auto& dim : dims)
		{
			pb_ten.add_dims(dim.dim_value());
		}
		created_tens.emplace(id,
			unmarshaler.unmarsh_leaf(pb_ten, teq::Placeholder, name));
	}

	const auto& pb_tens = pb_graph.initializer();
	for (const TensorProto& pb_ten : pb_tens)
	{
		std::string id = pb_ten.name();
		std::string name;
		teq::Usage usage = teq::Unknown;
		if (estd::has(annotations, id))
		{
			AnnotationsT& ans = annotations[id];
			name = estd::try_get(ans, leafname_key, "");
			usage = teq::get_named_usage(
				estd::try_get(ans, leafusage_key, ""));
		}
		created_tens.emplace(id,
			unmarshaler.unmarsh_leaf(pb_ten, usage, name));
	}

	const auto& pb_nodes = pb_graph.node();
	for (const NodeProto& pb_node : pb_nodes)
	{
		assert(pb_node.has_op_type());
		std::string id = pb_node.name();
		std::string opname = pb_node.op_type();
		marsh::Maps attrs;
		const auto& pb_attrs = pb_node.attribute();
		const auto& inputs = pb_node.input();
		teq::TensptrsT args;
		args.reserve(inputs.size());
		std::transform(inputs.begin(), inputs.end(), std::back_inserter(args),
			[&created_tens](std::string input)
			{
				return estd::must_getf(created_tens, input,
					"failed to find %s", input.c_str());
			});
		teq::TensptrT tens;
		if (const GraphProto* sub = unmarshal_attrs(attrs, pb_attrs))
		{
			teq::TensptrsT roots;
			load_graph(roots, *sub, unmarshaler, created_tens);
			tens = unmarshaler.unmarsh_layr(opname, roots, args, std::move(attrs));
		}
		else
		{
			tens = unmarshaler.unmarsh_func(opname, args, std::move(attrs));
		}
		created_tens.emplace(id, tens);
	}

	const auto& pb_outputs = pb_graph.output();
	for (const ValueInfoProto& pb_output : pb_outputs)
	{
		roots.push_back(created_tens.at(pb_output.name()));
	}
}

}

#endif
