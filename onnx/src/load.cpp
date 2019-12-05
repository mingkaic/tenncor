#include "onnx/load.hpp"

#ifdef ONNX_LOAD_HPP

namespace onnx
{

void load_graph (teq::TensptrsT& roots, const GraphProto& pb_graph,
	LeafUnmarshF unmarshal_leaf, FuncUnmarshF unmarshal_func)
{
	const auto& pb_annotations = pb_graph.quantization_annotation();

	auto annotations = unmarshal_annotation(pb_annotations);

	std::unordered_map<std::string,teq::TensptrT> generated_tens;
	const auto& pb_tens = pb_graph.initializer();
	for (const TensorProto& pb_ten : pb_tens)
	{
		std::string id = pb_ten.name();
		std::string name;
		bool is_const = false;
		if (estd::has(annotations, id))
		{
			AnnotationsT& ans = annotations[id];
			name = estd::try_get(ans, leafname_key, "");
			is_const = estd::has(ans, leafconst_key);
		}
		auto tens = unmarshal_leaf(pb_ten, is_const, name);
		generated_tens.emplace(id, tens);
	}

	// const auto& pb_inputs = pb_graph.input();
	// todo: distinguish weight ahd placeholders and set ph as inputs

	const auto& pb_nodes = pb_graph.node();
	for (const NodeProto& pb_node : pb_nodes)
	{
		assert(pb_node.has_op_type());
		std::string id = pb_node.name();
		std::string opname = pb_node.op_type();
		marsh::Maps attrs;
		const auto& pb_attrs = pb_node.attribute();
		unmarshal_attrs(attrs, pb_attrs);

		const auto& inputs = pb_node.input();
		teq::TensptrsT args;
		args.reserve(inputs.size());
		std::transform(inputs.begin(), inputs.end(), std::back_inserter(args),
			[&generated_tens](std::string input)
			{
				return generated_tens.at(input);
			});
		teq::TensptrT tens = unmarshal_func(
			opname, args, std::move(attrs));
		generated_tens.emplace(id, tens);
	}

	const auto& pb_outputs = pb_graph.output();
	for (const ValueInfoProto& pb_output : pb_outputs)
	{
		roots.push_back(generated_tens.at(pb_output.name()));
	}
}

}

#endif
