#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "ade/ileaf.hpp"
#include "ade/ifunctor.hpp"

#include "dbg/stream/ade.hpp"

#ifndef DBG_ADE_CSV_HPP
#define DBG_ADE_CSV_HPP

const char label_delim = ':';

static void multiline_replace (std::string& multiline)
{
	size_t i = 0;
	char nline = '\n';
	while ((i = multiline.find(nline, i)) != std::string::npos)
	{
		multiline.replace(i, 1, "\\");
	}
}

enum NODE_TYPE
{
	VARIABLE = 0,
	FUNCTOR,
	CACHED_FUNC,
};

using GetTypeFuncT = std::function<NODE_TYPE(ade::iFunctor*)>;

struct CSVEquation final : public ade::iTraveler
{
	CSVEquation (GetTypeFuncT get_ftype =
		[](ade::iFunctor* func) { return FUNCTOR; }) :
		get_ftype_(get_ftype) {}

	void visit (ade::iLeaf* leaf) override
	{
		if (nodes_.end() != nodes_.find(leaf))
		{
			return;
		}
		std::string label;
		auto it = labels_.find(leaf);
		if (labels_.end() != it)
		{
			label = it->second + "=";
		}
		label += leaf->to_string();
		if (showshape_)
		{
			label += leaf->shape().to_string();
		}
		nodes_.emplace(leaf, Node{
			label,
			VARIABLE,
			nodes_.size(),
		});
	}

	void visit (ade::iFunctor* func) override
	{
		if (nodes_.end() != nodes_.find(func))
		{
			return;
		}
		std::string funcstr;
		auto it = labels_.find(func);
		if (labels_.end() != it)
		{
			funcstr = it->second + "=";
		}
		funcstr += func->to_string();
		if (showshape_)
		{
			funcstr += func->shape().to_string();
		}
		nodes_.emplace(func, Node{
			funcstr,
			get_ftype_(func),
			nodes_.size(),
		});
		auto& children = func->get_children();
		for (size_t i = 0, n = children.size(); i < n; ++i)
		{
			const ade::FuncArg& child = children[i];
			auto coorder = child.get_coorder().get();
			auto tens = child.get_tensor().get();
			if (ade::is_identity(coorder))
			{
				coorder = nullptr;
			}
			else
			{
				std::string coordstr = coorder->to_string();
				multiline_replace(coordstr);
				coorders_.emplace(coorder, coordstr);
			}
			edges_.push_back(Edge{
				func,
				tens,
				coorder,
				fmts::to_string(i),
			});
			tens->accept(*this);
		}
	}

	void to_stream (std::ostream& out)
	{
		size_t nnodes = nodes_.size();
		for (size_t i = 0, nedges = edges_.size(); i < nedges; ++i)
		{
			const Edge& edge = edges_[i];
			auto& parent_node = nodes_[edge.func_];
			auto& child_node = nodes_[edge.child_];
			std::string color = child_node.ntype_ == CACHED_FUNC ?
				"red" : "white";
			if (nullptr == edge.coorder_)
			{
				out << parent_node.id_ << label_delim
					<< parent_node.label_ << ','
					<< child_node.id_ << label_delim
					<< child_node.label_ << ','
					<< edge.edge_label_ << ','
					<< color << '\n';
			}
			else
			{
				out << parent_node.id_ << label_delim
					<< parent_node.label_ << ','
					<< nnodes + i << label_delim
					<< coorders_[edge.coorder_] << ','
					<< edge.edge_label_ << ','
					<< color << '\n';

				out << nnodes + i << label_delim
					<< coorders_[edge.coorder_] << ','
					<< child_node.id_ << label_delim
					<< child_node.label_ << ','
					<< edge.edge_label_ << ','
					<< color << '\n';
			}
		}
	}

	bool showshape_ = false;

	struct Edge
	{
		ade::iFunctor* func_;

		ade::iTensor* child_;

		ade::iCoordMap* coorder_;

		std::string edge_label_;
	};

	struct Node
	{
		std::string label_;

		NODE_TYPE ntype_;

		size_t id_;
	};

	/// For every label associated with a tensor, show LABEL=value in the tree
	LabelsMapT labels_;

	std::vector<Edge> edges_;

	std::unordered_map<ade::iTensor*,Node> nodes_;

	std::unordered_map<ade::iCoordMap*,std::string> coorders_;

	GetTypeFuncT get_ftype_;
};

#endif // DBG_ADE_CSV_HPP
