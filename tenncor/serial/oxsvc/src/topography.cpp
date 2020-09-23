#include "internal/global/global.hpp"

#include "tenncor/serial/oxsvc/topography.hpp"

#ifdef DISTR_OX_TOPOGRAPHY_HPP

namespace distr
{

namespace ox
{

SegmentsT split_topograph (
	const onnx::GraphProto& graph,
	const TopographyT& topography)
{
	if (topography.empty())
	{
		global::fatal("topographic map cannot be empty");
	}

	types::StrUMapT<NodeT> nodes;
	const auto& pb_nodes = graph.node();
	for (const auto& pb_node : pb_nodes)
	{
		auto id = pb_node.name();
		auto color = estd::try_get(topography, id, "");
		nodes.emplace(id,
			std::make_shared<TopographicNode>(&pb_node, nodes, color));
	}

	NodeT node;
	auto& roots = graph.output();
	types::StrUMapT<NodesT> seginputs;
	for (auto& root : roots)
	{
		std::string rid = root.name();
		if (estd::get(node, nodes, rid))
		{
			if (node->color_.empty())
			{
				global::fatalf("root %s is not marked in "
					"topographic map", rid.c_str());
			}
			seginputs[node->color_].push_back(node);
		}
	}

	SegmentsT out;
	for (auto& seginput : seginputs)
	{
		out.push_back(std::make_shared<TopographicSeg>(
			graph, seginput.second));
	}
	return out;
}

}

}

#endif
