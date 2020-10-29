
#include "tenncor/layr.hpp"

#ifdef TENNCOR_LAYR_HPP

namespace tcr
{

eteq::ETensor connect (const eteq::ETensor& root, const eteq::ETensor& input)
{
	const global::CfgMapptrT& ctx = root.get_context();
	if (auto mgr = get_distrmgr(ctx))
	{
		// // get remote input

		// // serialize the remote graph
		// onnx::GraphProto pb_graph;
		// auto& oxsvc = distr::get_oxsvc(*mgr);
		// oxsvc.save_graph(pb_graph, {(teq::TensptrT) root});

		// // reload the graph replacing input
		// onnx::TensptrIdT replacements;
		// // replacements.insert({});
		// auto outs = oxsvc.load_graph(replacements, pb_graph);
		// if (outs.size() != 1)
		// {
		// 	global::info("failed to deserialize during layer connection");
		// }
		// return outs.front();
	}
	return layr::connect(root, input);
}

}

#endif
