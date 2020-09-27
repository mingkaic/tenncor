
#ifndef DISTR_OX_TOPOGRAPHY_HPP
#define DISTR_OX_TOPOGRAPHY_HPP

#include "tenncor/serial/oxsvc/util.hpp"

namespace distr
{

namespace ox
{

struct iTopographicNode;

struct TopographicSeg;

using NodeT = std::shared_ptr<iTopographicNode>;

using NodesT = std::vector<NodeT>;

using SegmentT = std::shared_ptr<TopographicSeg>;

using SegmentsT = std::list<SegmentT>;

using GraphT = types::StrUMapT<NodeT>;

struct iTopographicNode
{
	virtual ~iTopographicNode (void) = default;

	virtual std::string get_name (void) const = 0;

	virtual void contribute (onnx::GraphProto& graph) const = 0;

	std::string color_;

	std::unordered_set<NodeT> edges_;
};

struct TopographicOp final : public iTopographicNode
{
	TopographicOp (const onnx::NodeProto* source,
		const GraphT& existing_nodes) : source_(source)
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

	std::string get_name (void) const override
	{
		return source_->name();
	}

	void contribute (onnx::GraphProto& graph) const override
	{
		graph.add_node()->MergeFrom(*source_);
	}

	const onnx::NodeProto* source_;
};

struct TopographicInit final : public iTopographicNode
{
	TopographicInit (const onnx::TensorProto* init) : source_(init) {}

	std::string get_name (void) const override
	{
		return source_->name();
	}

	void contribute (onnx::GraphProto& graph) const override
	{
		graph.add_initializer()->MergeFrom(*source_);
	}

	const onnx::TensorProto* source_;
};

struct TopographicInput final : public iTopographicNode
{
	TopographicInput (const onnx::ValueInfoProto* input) : source_(input) {}

	std::string get_name (void) const override
	{
		return source_->name();
	}

	void contribute (onnx::GraphProto& graph) const override
	{
		graph.add_input()->MergeFrom(*source_);
	}

	const onnx::ValueInfoProto* source_;
};

struct TopographicSeg final
{
	TopographicSeg (const onnx::GraphProto& graph, const NodesT& nodes) :
		color_(nodes.front()->color_)
	{
		merge_graph_proto(graph_, graph, {NODE, INIT, INPUT});

		for (const auto& node : nodes)
		{
			auto id = node->get_name();
			graph_.add_output()->set_name(id);
		}

		types::StrUMapT<NodesT> refs;
		std::list<NodeT> q(nodes.begin(), nodes.end());
		std::list<NodeT> botup;
		std::unordered_set<iTopographicNode*> visited;
		while (false == q.empty())
		{
			auto node = q.front();
			q.pop_front();
			if (estd::has(visited, node.get()))
			{
				continue;
			}
			visited.emplace(node.get());
			if (node->color_.size() > 0 && node->color_ != color_)
			{
				refs[node->color_].push_back(node);
			}
			else
			{
				botup.push_front(node);
				q.insert(q.end(), node->edges_.begin(), node->edges_.end());
			}
		}
		// add graph nodes in bottom-top order
		for (auto node : botup)
		{
			node->contribute(graph_);
		}

		for (const auto& ref : refs)
		{
			subgraphs_.emplace(ref.first,
				std::make_shared<TopographicSeg>(graph, ref.second));
		}
	}

	std::string color_;

	onnx::GraphProto graph_;

	types::StrUMapT<SegmentT> subgraphs_;
};

void extract_nodes (
	GraphT& out, const onnx::GraphProto& graph);

SegmentsT split_topograph (
	const onnx::GraphProto& graph,
	const TopographyT& topography);

}

}

#endif // DISTR_OX_TOPOGRAPHY_HPP
