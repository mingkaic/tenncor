
#ifndef DISTR_OX_TOPOGRAPHY_HPP
#define DISTR_OX_TOPOGRAPHY_HPP

#include "tenncor/serial/oxsvc/util.hpp"

namespace distr
{

namespace ox
{

struct TopographicNode;

struct TopographicSeg;

using NodeT = std::shared_ptr<TopographicNode>;

using NodesT = std::vector<NodeT>;

using SegmentT = std::shared_ptr<TopographicSeg>;

using SegmentsT = std::list<SegmentT>;

struct TopographicNode
{
	TopographicNode (const onnx::NodeProto* source,
		const types::StrUMapT<NodeT>& existing_nodes,
		const std::string& color) : source_(source), color_(color)
	{
		const auto& inputs = source->input();
		const auto& attrs = source->attribute();
		for (const auto& input : inputs)
		{
			if (auto node = estd::try_get(existing_nodes, input, nullptr))
			{
				edges_.emplace(node);
			}
		}
		for (const auto& attr : attrs)
		{
			switch (attr.type())
			{
				case onnx::AttributeProto::TENSOR:
					if (auto node = estd::try_get(existing_nodes, attr.t().name(), nullptr))
					{
						edges_.emplace(node);
					}
					break;
				case onnx::AttributeProto::TENSORS:
					for (const auto& tensor : attr.tensors())
					{
						if (auto node = estd::try_get(existing_nodes, tensor.name(), nullptr))
						{
							edges_.emplace(node);
						}
					}
					break;
				default:
					break;
			}
		}
	}

	const onnx::NodeProto* source_;

	std::string color_;

	std::unordered_set<NodeT> edges_;
};

struct TopographicSeg final
{
	TopographicSeg (const onnx::GraphProto& graph, const NodesT& nodes) :
		color_(nodes.front()->color_)
	{
		merge_graph_proto(graph_, graph, {NODE});

		types::StrUMapT<NodesT> refs;
		std::list<NodeT> q(nodes.begin(), nodes.end());
		std::list<const onnx::NodeProto*> sources;
		while (false == q.empty())
		{
			auto node = q.front();
			q.pop_front();
			if (node->color_.size() > 0 && node->color_ != color_)
			{
				refs[node->color_].push_back(node);
			}
			else
			{
				sources.push_front(node->source_);
			}
			q.insert(q.end(), node->edges_.begin(), node->edges_.end());
		}
		// add graph nodes in bottom-top order
		for (auto source : sources)
		{
			graph_.add_node()->MergeFrom(*source);
		}

		for (const auto& ref : refs)
		{
			subgraphs_.emplace(ref.first,
				std::make_shared<TopographicSeg>(graph, ref.second));
		}

		for (const auto& node : nodes)
		{
			auto id = node->source_->name();
			graph_.add_output()->set_name(id);
		}
	}

	std::string color_;

	onnx::GraphProto graph_;

	types::StrUMapT<SegmentT> subgraphs_;
};

SegmentsT split_topograph (
	const onnx::GraphProto& graph,
	const TopographyT& topography);

}

}

#endif // DISTR_OX_TOPOGRAPHY_HPP
