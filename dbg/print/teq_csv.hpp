///
/// teq_csv.hpp
/// dbg
///
/// Purpose:
/// Draw an equation graph edges in CSV format
///

#ifndef DBG_TEQ_CSV_HPP
#define DBG_TEQ_CSV_HPP

#include <utility>

#include "dbg/print/teq.hpp"

/// CSV delimiter
const char label_delim = ':';

/// Type of the tensors
enum NODE_TYPE
{
	VARIABLE = 0,
	FUNCTOR,
	CACHED_FUNC,
};

/// Function that identify functors by NODE_TYPE
using GetTypeF = std::function<NODE_TYPE(teq::iFunctor&)>;

/// Use CSVEquation to render teq::TensptrT graph to output csv edges
struct CSVEquation final : public teq::iOnceTraveler
{
	CSVEquation (GetTypeF get_ftype =
		[](teq::iFunctor&) { return FUNCTOR; }) :
		get_ftype_(get_ftype) {}

	/// Stream visited graphs to out
	void to_stream (std::ostream& out)
	{
		size_t nedges = edges_.size();
		for (size_t i = 0; i < nedges; ++i)
		{
			const Edge& edge = edges_[i];
			auto& parent_node = nodes_[edge.func_];
			auto& child_node = nodes_[edge.child_];
			std::string color = child_node.ntype_ == CACHED_FUNC ?
				"red" : "white";

			out << parent_node.id_ << label_delim << parent_node.label_ << ','
				<< child_node.id_ << label_delim << child_node.label_ << ','
				<< edge.edge_label_ << ','
				<< color << '\n';
		}
	}

	/// Print every tensor's shape if true, otherwise don't
	bool showshape_ = false;

	teq::TensMapT<std::string> abbreviate_;

	/// For every label associated with a tensor, show LABEL=value in the tree
	LabelsMapT labels_;

private:
	/// Implementation of iTraveler
	void visit_leaf (teq::iLeaf& leaf) override
	{
		std::string label;
		auto it = labels_.find(&leaf);
		if (labels_.end() != it)
		{
			label = it->second + "=";
		}
		label += leaf.to_string();
		if (showshape_)
		{
			label += leaf.shape().to_string();
		}
		nodes_.emplace(&leaf, Node{
			label,
			VARIABLE,
			nodes_.size(),
		});
	}

	/// Implementation of iTraveler
	void visit_func (teq::iFunctor& func) override
	{
		std::string abbrev;
		if (estd::get(abbrev, abbreviate_, &func))
		{
			if (showshape_)
			{
				abbrev += func.shape().to_string();
			}
			nodes_.emplace(&func, Node{abbrev, VARIABLE, nodes_.size()});
			return;
		}
		std::string funcstr;
		if (estd::get(funcstr, labels_, &func))
		{
			funcstr += "=";
		}
		funcstr += func.to_string();
		if (showshape_)
		{
			funcstr += func.shape().to_string();
		}
		nodes_.emplace(&func, Node{funcstr, get_ftype_(func), nodes_.size()});
		auto args = func.get_args();
		for (size_t i = 0, nargs = args.size(); i < nargs; ++i)
		{
			edges_.push_back(Edge{&func, args[i].get(), fmts::to_string(i)});
		}
		teq::multi_visit(*this, args);
	}

	struct Edge
	{
		teq::iFunctor* func_;

		teq::iTensor* child_;

		std::string edge_label_;
	};

	struct Node
	{
		std::string label_;

		NODE_TYPE ntype_;

		size_t id_;
	};

	std::vector<Edge> edges_;

	teq::TensMapT<Node> nodes_;

	GetTypeF get_ftype_;
};

#endif // DBG_TEQ_CSV_HPP
