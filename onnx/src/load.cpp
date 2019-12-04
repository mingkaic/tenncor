#include "onnx/load.hpp"

#ifdef ONNX_LOAD_HPP

namespace onnx
{

void load_graph (teq::TensptrSetT& roots, const GraphProto& pb_graph,
	LeafUnmarshF unmarshal_leaf, FuncUnmarshF unmarshal_func)
{
	auto pb_nodes = pb_graph.node();
	auto pb_tens = pb_graph.initializer();
	// const auto& pb_inputs = pb_graph.input();
	// const auto& pb_outputs = pb_graph.output();
	assert(pb_tens.size() == pb_nodes.size());
	std::unordered_map<std::string,teq::TensptrT> generated_tens;
	for (size_t i = 0, n = pb_tens.size(); i < n; ++i)
	{
		const NodeProto& pb_node = pb_nodes[i];
		teq::TensptrT tens;
		if (pb_node.has_op_type())
		{
			std::string opname = pb_node.op_type();
			marsh::Maps attrs;
			const auto& pb_attrs = pb_node.attribute();
			unmarshal_attrs(attrs, pb_attrs);

			const auto& inputs = pb_node.input();
			auto dims = pb_tens[i].dims();
			std::vector<teq::DimT> slist(dims.begin(), dims.end());
			EdgesT args;
			for (std::string input : inputs)
			{
				// const ValueInfoProto& pb_input = pb_inputs.at(input);
				teq::TensptrT ctens = generated_tens[input];
				args.push_back({ctens, teq::Shape(slist)});
				roots.erase(ctens);
			}
			tens = unmarshal_func(opname, args, std::move(attrs));
		}
		else
		{
			marsh::Maps attrs;
			const auto& pb_attrs = pb_node.attribute();
			unmarshal_attrs(attrs, pb_attrs);
			tens = unmarshal_leaf(pb_tens[i],
				estd::has(attrs.contents_, leafconst_key));
		}
		// const auto& outputs = pb_node.output();
		generated_tens.emplace(pb_node.name(), tens);
		roots.emplace(tens);
	}
}

}

#endif
