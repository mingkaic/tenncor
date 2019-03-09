#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "ade/ileaf.hpp"
#include "ade/ifunctor.hpp"

const char label_delim = ':';

void multiline_replace (std::string& multiline);

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
		if (visited_.end() != visited_.find(leaf))
		{
			return;
		}
		nodes_.emplace(leaf, Node{
			leaf->to_string(),
			VARIABLE,
			nodes_.size(),
		});
		visited_.emplace(leaf);
	}

	void visit (ade::iFunctor* func) override
	{
		if (visited_.end() != visited_.find(func))
		{
			return;
		}
		std::string funcstr = func->to_string();
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
				i
			});
			tens->accept(*this);
		}
		visited_.emplace(func);
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
					<< edge.child_idx_ << ','
					<< color << '\n';
			}
			else
			{
				out << parent_node.id_ << label_delim
					<< parent_node.label_ << ','
					<< nnodes + i << label_delim
					<< coorders_[edge.coorder_] << ','
					<< edge.child_idx_ << ','
					<< color << '\n';

				out << nnodes + i << label_delim
					<< coorders_[edge.coorder_] << ','
					<< child_node.id_ << label_delim
					<< child_node.label_ << ','
					<< edge.child_idx_ << ','
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

		size_t child_idx_;
	};

	struct Node
	{
		std::string label_;

		NODE_TYPE ntype_;

		size_t id_;
	};

	std::vector<Edge> edges_;

	std::unordered_map<ade::iTensor*,Node> nodes_;

	std::unordered_map<ade::iCoordMap*,std::string> coorders_;

	std::unordered_set<ade::iTensor*> visited_;

	GetTypeFuncT get_ftype_;
};
