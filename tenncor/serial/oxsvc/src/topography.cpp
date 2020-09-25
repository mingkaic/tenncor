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
	const auto& pb_inputs = graph.input();
	const auto& pb_inits = graph.initializer();
	const auto& pb_nodes = graph.node();
	for (const auto& pb_input : pb_inputs)
	{
		auto id = pb_input.name();
		auto color = estd::try_get(topography, id, "");
		nodes.emplace(id,
			std::make_shared<TopographicInput>(&pb_input, color));
	}
	for (const auto& pb_init : pb_inits)
	{
		auto id = pb_init.name();
		auto color = estd::try_get(topography, id, "");
		nodes.emplace(id,
			std::make_shared<TopographicInit>(&pb_init, color));
	}
	for (const auto& pb_node : pb_nodes)
	{
		auto id = pb_node.name();
		auto color = estd::try_get(topography, id, "");
		nodes.emplace(id,
			std::make_shared<TopographicNode>(&pb_node, nodes, color));
	}

	NodeT node;
	auto& roots = graph.output();
	types::StrUMapT<NodesT> segroots;
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
			segroots[node->color_].push_back(node);
		}
	}

	SegmentsT out;
	for (auto& segroot : segroots)
	{
		out.push_back(std::make_shared<TopographicSeg>(
			graph, segroot.second));
	}
	return out;
}

}

}

#endif
