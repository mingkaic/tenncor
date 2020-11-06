
#include "tenncor/layr.hpp"

#ifdef TENNCOR_LAYR_HPP

namespace tcr
{

const std::string root_id = "root";

const std::string input_id = "layer_input";

static teq::TensptrT lookup_input (teq::TensptrT root, distr::iDistrManager& mgr)
{
	auto& lusvc = distr::get_lusvc(mgr);

	std::string root_str;
	if (auto root_ref = dynamic_cast<distr::iDistrRef*>(root.get()))
	{
		root_str = root_ref->remote_string();
	}
	else
	{
		root_str = root->to_string();
	}
	std::stringstream condjson;
	condjson << "{"
		"\"op\":{"
			"\"opname\":\"" << root_str << "\","
			"\"attrs\":{"
				"\"" << teq::layer_attr << "\":{"
					"\"layer\":{"
						"\"input\":{"
							"\"symb\":\"input\""
						"}"
					"}"
				"}"
			"}"
		"}"
	"}";
	query::Node input_cond;
	query::json_parse(input_cond, condjson);

	auto detections = lusvc.query({root}, input_cond);
	for (auto detection : detections)
	{
		if (detection.root_ == root)
		{
			return detection.symbs_.at("input");
		}
	}
	return nullptr;
}

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

eteq::ETensor connect (const eteq::ETensor& root, const eteq::ETensor& input)
{
	const global::CfgMapptrT& ctx = root.get_context();
	if (auto mgr = get_distrmgr(ctx))
	{
		auto rinput = lookup_input(root, *mgr);
		if (nullptr == rinput)
		{
			global::fatal("failed to find layer input");
		}
		auto leaf_refs = lookup_leaves(root, *mgr);

		auto& oxsvc = distr::get_oxsvc(*mgr);

		// serialize the remote graph
		onnx::TensptrIdT ids;
		onnx::GraphProto pb_graph;
		ids.insert({root, root_id});
		ids.insert({rinput, input_id});
		for (size_t i = 0, n = leaf_refs.size(); i < n; ++i)
		{
			ids.insert({leaf_refs[i], fmts::to_string(i)});
		}
		oxsvc.save_graph(pb_graph, {(teq::TensptrT) root}, ids, {rinput.get()});

		// reload the graph replacing input
		onnx::TensptrIdT replacements;
		replacements.insert({(teq::TensptrT) input, input_id});
		for (size_t i = 0, n = leaf_refs.size(); i < n; ++i)
		{
			replacements.insert({leaf_refs[i], fmts::to_string(i)});
		}
		auto outs = oxsvc.load_graph(replacements, pb_graph);
		if (outs.size() != 1)
		{
			global::fatal("failed to deserialize during layer connection");
		}
		return eteq::ETensor(outs.front(), ctx);
	}
	return layr::connect(root, input);
}

}

#endif
