#ifndef QUERY_QUERY_HPP
#define QUERY_QUERY_HPP

#include <boost/functional/hash.hpp>

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

inline bool equals (const teq::iLeaf* leaf, const Leaf& var)
{
	if (nullptr == leaf)
	{
		return false;
	}
	return
		(Leaf::kLabel != var.nullable_label_case() ||
			var.label() == leaf->to_string()) &&
		(Leaf::kDtype != var.nullable_dtype_case() ||
			var.dtype() == leaf->type_label()) &&
		(Leaf::kUsage != var.nullable_usage_case() ||
			var.usage() == teq::get_usage_name(leaf->get_usage())) &&
		(0 == var.shape_size() ||
			to_shape(var.shape()).compatible_after(leaf->shape(), 0));
}

struct QueryResult final
{
	operator teq::iTensor*() const
	{
		return root_;
	}

	friend bool operator == (const QueryResult& a, const QueryResult& b)
	{
		return a.root_ == b.root_ && a.symbs_ == b.symbs_;
	}

	teq::iTensor* root_;

	SymbMapT symbs_;
};

struct QResHasher
{
	size_t operator ()(const QueryResult& res) const
	{
		size_t seed = 0;
        boost::hash_combine(seed, res.root_);
        boost::hash_combine(seed, boost::hash_range(
			res.symbs_.begin(), res.symbs_.end()));
		return seed;
	}
};

using QResultsT = std::vector<QueryResult>;

using QReSetT = std::unordered_set<QueryResult,QResHasher>;

using QAttrMapT = google::protobuf::Map<std::string,Attribute>;

bool equals (
	QResultsT& candidates, const marsh::iObject* attr,
	const Attribute& pba, const Query& matcher);

struct Query final : public teq::iOnceTraveler
{
	QResultsT match (const Node& cond) const
	{
		PathsT paths;
		switch (cond.val_case())
		{
			case Node::ValCase::kSymb:
				paths.reserve(visited_.size());
				std::transform(visited_.begin(), visited_.end(),
					std::back_inserter(paths),
					[](teq::iTensor* node)
					{
						return std::make_shared<Path>(node);
					});
				symb_filter(paths, cond.symb());
				break;
			case Node::ValCase::kCst:
				paths.reserve(visited_.size());
				std::transform(visited_.begin(), visited_.end(),
					std::back_inserter(paths),
					[](teq::iTensor* node)
					{
						return std::make_shared<Path>(node);
					});
				leaf_filter(paths, cond.cst());
				break;
			case Node::ValCase::kLeaf:
				paths.reserve(visited_.size());
				std::transform(visited_.begin(), visited_.end(),
					std::back_inserter(paths),
					[](teq::iTensor* node)
					{
						return std::make_shared<Path>(node);
					});
				leaf_filter(paths, cond.leaf());
				break;
			case Node::ValCase::kOp:
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
		QReSetT existing_res;
		for (PathptrT path : paths)
		{
			QueryResult result{path->tens_, path->symbols_};
			if (false == estd::has(existing_res, result))
			{
				results.push_back(result);
				existing_res.emplace(result);
			}
		}
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

	bool surface_matches (QResultsT& tens, const teq::iFunctor* func,
		const Operator& cond, const QAttrMapT& pb_attrs) const
	{
		if (nullptr == func)
		{
			return false;
		}
		std::string cop = cond.opname();
		size_t ncargs = cond.args().size();
		return func->get_opcode().name_ == cop &&
			ncargs <= func->get_children().size() &&
			std::all_of(pb_attrs.begin(), pb_attrs.end(),
				[&](const auto& pb_attr)
				{
					return equals(tens, func->get_attr(pb_attr.first),
						pb_attr.second, *this);
				});
	}

	void noncomm_filter (PathsT& paths, const Operator& cond) const
	{
		for (const Node& carg : cond.args())
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

	void commutative_filter (PathsT& paths, const Operator& cond) const
	{
		const auto& cargs = cond.args();
		size_t nargs = cargs.size();
		for (const Node& carg : cargs)
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

	void attr_filter (PathsT& paths, const Operator& cond) const
	{
		const QAttrMapT& attrs = cond.attrs();
		PathsT mpaths;
		for (const PathptrT& root : paths)
		{
			QResultsT attr_matches;
			if (surface_matches(attr_matches,
				dynamic_cast<const teq::iFunctor*>(root->tens_), cond, attrs))
			{
				for (auto& attr_match : attr_matches)
				{
					teq::iTensor* tens;
					if (std::all_of(attr_match.symbs_.begin(), attr_match.symbs_.end(),
						[&](const std::pair<std::string,teq::iTensor*>& sympair)
						{
							return false == estd::get(
								tens, root->symbols_, sympair.first) ||
								tens == sympair.second;
						}))
					{
						PathptrT rclone = std::make_shared<Path>(*root);
						rclone->symbols_.merge(attr_match.symbs_);
						mpaths.push_back(rclone);
					}
				}
				if (attr_matches.empty())
				{
					mpaths.push_back(root);
				}
			}
		}
		paths = PathsT(mpaths.begin(), mpaths.end());
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

	void func_filter (PathsT& paths, const Operator& cond) const
	{
		attr_filter(paths, cond);
		if (egen::is_commutative(cond.opname()))
		{
			commutative_filter(paths, cond);
		}
		else
		{
			noncomm_filter(paths, cond);
		}
		if (Operator::kCapture == cond.nullable_capture_case())
		{
			symb_filter(paths, cond.capture());
		}
	}

	void match_helper (PathsT& paths, const Node& cond) const
	{
		if (paths.empty())
		{
			return;
		}
		switch (cond.val_case())
		{
			case Node::ValCase::kSymb:
				symb_filter(paths, cond.symb());
				break;
			case Node::ValCase::kCst:
				leaf_filter(paths, cond.cst());
				break;
			case Node::ValCase::kLeaf:
				leaf_filter(paths, cond.leaf());
				break;
			case Node::ValCase::kOp:
				func_filter(paths, cond.op());
				break;
			default:
				teq::fatal("cannot look for unknown node");
		}
	}
};

}

#endif // QUERY_QUERY_HPP
