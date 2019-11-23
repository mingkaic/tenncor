///
/// teq_csv.hpp
/// dbg
///
/// Purpose:
/// Draw an equation graph edges in CSV format
///

#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "teq/ileaf.hpp"
#include "teq/ifunctor.hpp"

#include "estd/estd.hpp"

#include "dbg/stream/teq.hpp"

#ifndef DBG_TEQ_CSV_HPP
#define DBG_TEQ_CSV_HPP

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
using GetTypeF = std::function<NODE_TYPE(teq::iFunctor*)>;

/// Use CSVEquation to render teq::TensptrT graph to output csv edges
struct CSVEquation final : public teq::iTraveler
{
	CSVEquation (GetTypeF get_ftype =
		[](teq::iFunctor* func) { return FUNCTOR; }) :
		get_ftype_(get_ftype) {}

	/// Implementation of iTraveler
	void visit (teq::iLeaf* leaf) override
	{
		if (estd::has(nodes_, leaf))
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

	/// Implementation of iTraveler
	void visit (teq::iFunctor* func) override
	{
		if (estd::has(nodes_, func))
		{
			return;
		}
		std::string abbrev;
		if (estd::get(abbrev, abbreviate_, func))
		{
			if (showshape_)
			{
				abbrev += func->shape().to_string();
			}
			nodes_.emplace(func, Node{
				abbrev,
				VARIABLE,
				nodes_.size(),
			});
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
		auto children = func->get_children();
		for (size_t i = 0, nchildren = children.size(); i < nchildren; ++i)
		{
			const teq::iEdge& child = children[i];
			auto tens = child.get_tensor().get();
			marsh::Maps mvalues;
			child.get_attrs(mvalues);
			edges_.push_back(Edge{
				func,
				tens,
				std::move(mvalues.contents_),
				fmts::to_string(i),
			});
			tens->accept(*this);
		}
	}

	/// Stream visited graphs to out
	void to_stream (std::ostream& out)
	{
		size_t nedges = edges_.size();
		std::unordered_set<std::string> attr_keys;
		for (size_t i = 0; i < nedges; ++i)
		{
			for (auto& apairs : edges_[i].attrs_)
			{
				attr_keys.emplace(apairs.first);
			}
		}
		std::vector<std::string> akeys(attr_keys.begin(), attr_keys.end());
		std::sort(akeys.begin(), akeys.end());

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
				<< color;
			for (std::string akey : akeys)
			{
				out << ',';
				if (estd::has(edge.attrs_, akey))
				{
					out << edge.attrs_.at(akey)->to_string();
				}
			}
			out << '\n';
		}
	}

	/// Print every tensor's shape if true, otherwise don't
	bool showshape_ = false;

	std::unordered_map<teq::iTensor*,std::string> abbreviate_;

	/// For every label associated with a tensor, show LABEL=value in the tree
	LabelsMapT labels_;

private:
	struct Edge
	{
		teq::iFunctor* func_;

		teq::iTensor* child_;

		std::unordered_map<std::string,marsh::ObjptrT> attrs_;

		std::string edge_label_;
	};

	struct Node
	{
		std::string label_;

		NODE_TYPE ntype_;

		size_t id_;
	};

	std::vector<Edge> edges_;

	std::unordered_map<teq::iTensor*,Node> nodes_;

	GetTypeF get_ftype_;
};

#endif // DBG_TEQ_CSV_HPP
