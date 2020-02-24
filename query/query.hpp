#ifndef QUERY_QUERY_HPP
#define QUERY_QUERY_HPP

#include "eigen/generated/opcode.hpp"

#include "query/query.pb.h"
#include "query/path.hpp"

namespace query
{

template <typename VEC, typename UNARY>
inline void remove_if (VEC& vec, UNARY pred)
{
	auto pend = std::remove_if(vec.begin(), vec.end(), pred);
	vec.erase(pend, vec.end());
}

using SindexT = std::unordered_map<std::string,teq::TensT>;

struct Query;

static inline teq::Shape to_shape (
	const google::protobuf::RepeatedField<uint32_t>& sfields)
{
	std::vector<teq::DimT> slist(sfields.begin(), sfields.end());
	return teq::Shape(slist);
}

inline bool equals (const teq::iLeaf* leaf, double scalar)
{
	if (nullptr == leaf)
	{
		return false;
	}
	return teq::IMMUTABLE == leaf->get_usage() &&
		leaf->to_string() == fmts::to_string(scalar);
}

inline bool equals (const teq::iLeaf* leaf, const query::Leaf& var)
{
	if (nullptr == leaf)
	{
		return false;
	}
	return
		(query::Leaf::kLabel != var.nullable_label_case() ||
			var.label() == leaf->to_string()) &&
		(query::Leaf::kDtype != var.nullable_dtype_case() ||
			var.dtype() == leaf->type_label()) &&
		(query::Leaf::kUsage != var.nullable_usage_case() ||
			var.usage() == teq::get_usage_name(leaf->get_usage())) &&
		(0 == var.shape_size() ||
			to_shape(var.shape()).compatible_after(leaf->shape(), 0));
}

bool equals (const marsh::iObject* attr,
	const query::Attribute& pba, const Query& matcher);

struct QueryResult final
{
	operator teq::iTensor*() const
	{
		return root_;
	}

	teq::iTensor* root_;

	SymbMapT symbs_;
};

using QResultsT = std::vector<QueryResult>;

struct Query final : public teq::iOnceTraveler
{
	QResultsT match (const query::Node& cond) const
	{
		PathsT paths;
		switch (cond.val_case())
		{
			case query::Node::ValCase::kSymb:
				paths.reserve(visited_.size());
				std::transform(visited_.begin(), visited_.end(),
					std::back_inserter(paths),
					[](teq::iTensor* node)
					{
						return std::make_shared<Path>(node);
					});
				symb_filter(paths, cond.symb());
				break;
			case query::Node::ValCase::kCst:
				paths.reserve(visited_.size());
				std::transform(visited_.begin(), visited_.end(),
					std::back_inserter(paths),
					[](teq::iTensor* node)
					{
						return std::make_shared<Path>(node);
					});
				leaf_filter(paths, cond.cst());
				break;
			case query::Node::ValCase::kLeaf:
				paths.reserve(visited_.size());
				std::transform(visited_.begin(), visited_.end(),
					std::back_inserter(paths),
					[](teq::iTensor* node)
					{
						return std::make_shared<Path>(node);
					});
				leaf_filter(paths, cond.leaf());
				break;
			case query::Node::ValCase::kOp:
			{
				teq::TensT nodes;
				if (estd::get(nodes, sindex_, cond.op().opname()))
				{
					paths.reserve(nodes.size());
					std::transform(nodes.begin(), nodes.end(),
						std::back_inserter(paths),
						[](teq::iTensor* node)
						{
							return std::make_shared<Path>(node);
						});
					match_helper(paths, cond);
				}
			}
				break;
			default:
				teq::fatal("cannot look for unknown node");
		}
		QResultsT results;
		std::transform(paths.begin(), paths.end(),
			std::back_inserter(results),
			[](PathptrT path)
			{
				return QueryResult{path->tens_, path->symbols_};
			});
		return results;
	}

	SindexT sindex_;

private:
	/// Implementation of iOnceTraveler
	void visit_leaf (teq::iLeaf& leaf) override {}

	/// Implementation of iOnceTraveler
	void visit_func (teq::iFunctor& func) override
	{
		sindex_[func.get_opcode().name_].push_back(&func);
		auto children = func.get_children();
		for (auto& arg : children)
		{
			arg->accept(*this);
		}
	}

	bool surface_matches (teq::iTensor* node, const query::Operator& cond,
		const google::protobuf::Map<std::string,query::Attribute>& pb_attrs) const
	{
		if (auto f = dynamic_cast<teq::iFunctor*>(node))
		{
			std::string cop = cond.opname();
			const auto& cargs = cond.args();
			return f->get_opcode().name_ == cop &&
				cargs.size() <= f->get_children().size() &&
				std::all_of(pb_attrs.begin(), pb_attrs.end(),
					[&](const auto& pb_attr)
					{
						return equals(f->get_attr(pb_attr.first), pb_attr.second, *this);
					});
		}
		return false;
	}

	void noncomm_filter (PathsT& paths, const query::Operator& cond) const
	{
		for (const query::Node& carg : cond.args())
		{
			PathsT children;
			children.reserve(paths.size());
			std::transform(paths.begin(), paths.end(),
				std::back_inserter(children),
				[](PathptrT& path)
				{
					auto args = path->get_args();
					assert(args.size() > 0);
					auto arg = args.front();
					return std::make_shared<Path>(arg.second,
						PrevT{arg.first, path}, path->symbols_);
				});
			match_helper(children, carg);
			paths.clear();
			std::transform(children.begin(), children.end(),
				std::back_inserter(paths),
				[](PathptrT match)
				{
					return match->recall();
				});
		}
	}

	void commutative_filter (PathsT& paths, const query::Operator& cond) const
	{
		const auto& cargs = cond.args();
		size_t nargs = cargs.size();
		for (const query::Node& carg : cargs)
		{
			PathsT children;
			children.reserve(paths.size() * nargs);
			for (auto& root : paths)
			{
				auto args = root->get_args();
				for (std::pair<size_t,teq::iTensor*>& arg : args)
				{
					children.push_back(std::make_shared<Path>(
						arg.second, PrevT{arg.first, root}, root->symbols_));
				}
			}
			match_helper(children, carg);
			paths.clear();
			std::transform(children.begin(), children.end(),
				std::back_inserter(paths),
				[](PathptrT match)
				{
					return match->recall();
				});
		}
		remove_if(paths,
			[&](PathptrT root)
			{
				return root->memory_.size() != nargs;
			});
	}

	void symb_filter (PathsT& paths, std::string cond) const
	{
		remove_if(paths,
			[&](PathptrT root)
			{
				return estd::has(root->symbols_, cond) &&
					root->symbols_[cond] != root->tens_;
			});
		for (PathptrT root : paths)
		{
			root->symbols_.emplace(cond, root->tens_);
		}
	}

	template <typename T>
	void leaf_filter (PathsT& paths, T leaf) const
	{
		remove_if(paths,
			[&](PathptrT root)
			{
				return false == equals(
					dynamic_cast<teq::iLeaf*>(root->tens_), leaf);
			});
	}

	void func_filter (PathsT& paths, const query::Operator& cond) const
	{
		const auto& attrs = cond.attrs();
		remove_if(paths,
			[&](PathptrT root)
			{
				return nullptr == dynamic_cast<teq::iFunctor*>(root->tens_) ||
					false == surface_matches(root->tens_, cond, attrs);
			});
		if (egen::is_commutative(cond.opname()))
		{
			commutative_filter(paths, cond);
		}
		else
		{
			noncomm_filter(paths, cond);
		}
		if (query::Operator::kCapture == cond.nullable_capture_case())
		{
			symb_filter(paths, cond.capture());
		}
	}

	void match_helper (PathsT& paths, const query::Node& cond) const
	{
		if (paths.empty())
		{
			return;
		}
		switch (cond.val_case())
		{
			case query::Node::ValCase::kSymb:
				symb_filter(paths, cond.symb());
				break;
			case query::Node::ValCase::kCst:
				leaf_filter(paths, cond.cst());
				break;
			case query::Node::ValCase::kLeaf:
				leaf_filter(paths, cond.leaf());
				break;
			case query::Node::ValCase::kOp:
				func_filter(paths, cond.op());
				break;
			default:
				teq::fatal("cannot look for unknown node");
		}
	}
};

}

#endif // QUERY_QUERY_HPP
