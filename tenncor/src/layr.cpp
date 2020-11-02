
#include "tenncor/layr.hpp"

#ifdef TENNCOR_LAYR_HPP

namespace tcr
{

const std::string root_id = "root";

static teq::TensptrsT lookup_leaves (teq::TensptrT root, distr::iDistrManager& mgr)
{
	auto& lusvc = distr::get_lusvc(mgr);

	std::stringstream condjson;
	condjson << "{\"leaf\":{}}";
	query::Node leaf_cond;
	query::json_parse(leaf_cond, condjson);

	auto detections = lusvc.query({root}, leaf_cond);
	return teq::TensptrsT(detections.begin(), detections.end());
}

static std::string get_input_id (const onnx::GraphProto& pb_graph)
{
	const onnx::NodeProto* root_node = nullptr;
	for (auto& pb_node : pb_graph.node())
	{
		if (root_id == pb_node.name())
		{
			root_node = &pb_node;
			break;
		}
	}
	if (nullptr == root_node)
	{
		global::fatal("root node not found");
	}
	std::string input_id;
	auto& pb_attrs = root_node->attribute();
	for (auto& pb_attr : pb_attrs)
	{
		if (pb_attr.type() == onnx::AttributeProto::GRAPH)
		{
			auto& inputs = pb_attr.g().output();
			if (inputs.size() != 1)
			{
				global::fatal("multiple inputs not supported");
			}
			input_id = inputs.begin()->name();
			break;
		}
	}
	if (input_id.empty())
	{
		global::fatal("input node not found");
	}
	return input_id;
}

static types::StringsT get_var_ids (const onnx::GraphProto& pb_graph)
{
	auto& inits = pb_graph.initializer();
	types::StringsT out;
	out.reserve(inits.size());
	for (auto& init : inits)
	{
		out.push_back(init.name());
	}
	return out;
}

eteq::ETensor connect (const eteq::ETensor& root, const eteq::ETensor& input)
{
	const global::CfgMapptrT& ctx = root.get_context();
	if (auto mgr = get_distrmgr(ctx))
	{
		auto& oxsvc = distr::get_oxsvc(*mgr);

		auto leaf_refs = lookup_leaves(root, *mgr);

		// serialize the remote graph
		onnx::TensptrIdT ids;
		onnx::GraphProto pb_graph;
		ids.insert({root, root_id});
		for (size_t i = 0, n = leaf_refs.size(); i < n; ++i)
		{
			ids.insert({leaf_refs[i], fmts::to_string(i)});
		}
		oxsvc.save_graph(pb_graph, {(teq::TensptrT) root}, ids, {root.get()});

		// get input id for replacement
		auto input_id = get_input_id(pb_graph);

		// get variable ids for replacement
		auto var_ids = get_var_ids(pb_graph);

		// reload the graph replacing input
		onnx::TensptrIdT replacements;
		replacements.insert({(teq::TensptrT) input, input_id});
		for (auto var_id : var_ids)
		{
			replacements.insert({ids.right.at(var_id), var_id});
		}
		auto outs = oxsvc.load_graph(replacements, pb_graph);
		if (outs.size() != 1)
		{
			global::info("failed to deserialize during layer connection");
		}
		return outs.front();
	}
	return layr::connect(root, input);
}

}

#endif
