//
/// query.hpp
/// query
///
/// Purpose:
/// Define subgraph filtering functionality
///

#ifndef QUERY_HPP
#define QUERY_HPP

#include "jobs/scope_guard.hpp"

#include "query/sindex.hpp"
#include "query/stats.hpp"
#include "query/parse.hpp"

namespace query
{

using SymbMapT = std::unordered_map<std::string,teq::iTensor*>;

struct QueryResult final
{
	teq::iTensor* root_;

	SymbMapT symbs_;
};

using QResultsT = std::vector<QueryResult>;

struct Query
{
	Query (search::OpTrieT& sindex) : from_(&sindex) {}

	// Return query with <existing symbol(s)> AND <symb>
	Query select (const std::string& symb)
	{
		// empty symbol denotes selecting roots, which is selected by default
		if (symb.empty())
		{
			return *this;
		}
		auto new_selections = selections_;
		new_selections.emplace(symb);
		return Query(
			from_,
			new_selections,
			conditions_);
	}

	// Return query with <existing condition> OR <condition>
	Query where (const ConditionT& condition)
	{
		auto new_conditions = conditions_;
		new_conditions.emplace(condition);
		return Query(
			from_,
			selections_,
			new_conditions);
	}

	Query where (std::istream& condition)
	{
		auto cond = std::make_shared<Node>();
		json_parse(*cond, condition);
		return where(cond);
	}

	// set depth_asc to true if max-distance from condition roots
	// to leaf/symbol and sum paths is sorted by min-max
	void exec (QResultsT& results, bool depth_asc = true)
	{
		StatsMapT order;
		auto root = from_->root();
		for (auto cond : conditions_)
		{
			SearchContext ctx;
			lookfor(ctx, root, *cond);
			for (auto& batch_pair : ctx.results_)
			{
				if (estd::has(order, batch_pair.first))
				{
					order[batch_pair.first] = std::max(
						order[batch_pair.first], batch_pair.second);
				}
				else
				{
					results.push_back(QueryResult{
						batch_pair.first,
						ctx.symbs_[batch_pair.first],
					});
					order.emplace(batch_pair);
				}
			}
		}
		std::sort(results.begin(), results.end(),
			[&](QueryResult& a, QueryResult& b)
			{
				if (depth_asc)
				{
					return order[a.root_] < order[b.root_];
				}
				return order[a.root_] > order[b.root_];
			});
	}

private:
	Query (search::OpTrieT* from,
		const std::unordered_set<std::string>& selections,
		const std::unordered_set<ConditionT>& conditions) :
		from_(from), selections_(selections), conditions_(conditions) {}

	struct SearchContext final
	{
		SearchContext (void) = default;

		SearchContext (const search::PathListT& path,
			const teq::TensSetT& blacklist) :
			path_(path), blacklist_(blacklist) {}

		search::PathListT path_;

		teq::TensSetT blacklist_;

		StatsMapT results_;

		teq::TensMapT<SymbMapT> symbs_;
	};

	// Return true if there's at least one result
	void lookfor (SearchContext& ctx, const search::OpTrieT::NodeT* tri,
		const Node& cond) // todo: add filter_out mechanism instead of allocing a subresult at every branch
	{
		if (nullptr == tri)
		{
			ctx.results_.clear();
			return;
		}
		switch (cond.val_case())
		{
			case Node::ValCase::kCst:
			{
				if (false == tri->leaf_.has_value())
				{
					ctx.results_.clear();
					return;
				}
				bool nomatch = true;
				double scalar = cond.cst();
				Stats stats{ctx.path_.size(), ctx.path_.size()};
				for (const auto& lpair : tri->leaf_->leaves_)
				{
					if (::query::equals(scalar, lpair.first))
					{
						nomatch = false;
						merge_stats(ctx.results_, lpair.second, stats);
					}
				}
				if (nomatch)
				{
					ctx.results_.clear();
				}
			}
				break;
			case Node::ValCase::kVar:
			{
				if (false == tri->leaf_.has_value())
				{
					ctx.results_.clear();
					return;
				}
				bool nomatch = true;
				const Variable& var = cond.var();
				Stats stats{ctx.path_.size(), ctx.path_.size()};
				for (const auto& lpair : tri->leaf_->leaves_)
				{
					if (::query::equals(var, lpair.first))
					{
						nomatch = false;
						merge_stats(ctx.results_, lpair.second, stats);
					}
				}
				if (nomatch)
				{
					ctx.results_.clear();
				}
			}
				break;
			case Node::ValCase::kOp:
			{
				const Operator& op = cond.op();
				const auto& attrs = op.attrs();
				if (attrs.size() > 0)
				{
					auto lookahead = static_cast<const search::OpTrieT::NodeT*>(
						tri->next(PathNode{0, egen::get_op(op.opname())}));
					if (false == lookahead->leaf_.has_value() ||
						lookahead->leaf_->attrs_.empty())
					{
						ctx.results_.clear();
						return;
					}
					teq::FuncMapT<teq::TensSetT> attr_matches;
					for (const auto& apair : lookahead->leaf_->attrs_)
					{
						teq::iFunctor* iattr = apair.first;
						std::unordered_set<std::string> need_keys;
						for (const auto& apair : attrs)
						{
							need_keys.emplace(apair.first);
						}
						for (auto jt = need_keys.begin(), et = need_keys.end();
							jt != et;)
						{
							std::string key = *jt;
							auto val = iattr->get_attr(key);
							if (nullptr != val && equals(attrs.at(key), val))
							{
								jt = need_keys.erase(jt);
							}
							else
							{
								++jt;
							}
						}
						if (need_keys.empty())
						{
							attr_matches.emplace(iattr, apair.second);
						}
					}
					// match the rest of the condition subgraph from trie root
					// in order to filter for matching attributable functors
					SearchContext subctx(ctx.path_, ctx.blacklist_);
					iterate_condition(subctx, from_->root(), op);
					ctx.blacklist_ = subctx.blacklist_;
					ctx.symbs_ = subctx.symbs_;
					// get attr_matches[subresults intersection attr_matches.keys]
					if (subctx.results_.empty())
					{
						ctx.results_.clear();
					}
					for (const auto& apair : attr_matches)
					{
						if (estd::has(subctx.results_, apair.first))
						{
							merge_stats(ctx.results_, apair.second,
								subctx.results_[apair.first]);
						}
					}
				}
				else
				{
					iterate_condition(ctx, tri, op);
				}
			}
				break;
			case Node::ValCase::kSymb:
				any_condition(ctx, cond.symb(), tri);
				break;
			default:
				teq::fatal("cannot look for unknown node");
		}
	}

	// Return true if there's at least one result
	void iterate_condition (SearchContext& ctx, const search::OpTrieT::NodeT* tri,
		const Operator& op)
	{
		egen::_GENERATED_OPCODE opcode = egen::get_op(op.opname());
		const auto& args = op.args();
		{
			ctx.path_.push_back(PathNode{0, opcode});
			jobs::ScopeGuard defer([&ctx] { ctx.path_.pop_back(); });
			if (args.empty())
			{
				any_condition(ctx, "", static_cast<const search::OpTrieT::NodeT*>(
					tri->next(PathNode{0, opcode})));
				return;
			}
			lookfor(ctx, static_cast<const search::OpTrieT::NodeT*>(
				tri->next(PathNode{0, opcode})), args[0]);
		}
		for (size_t i = 1, n = args.size(); i < n && false == ctx.results_.empty(); ++i)
		{
			SearchContext subctx(ctx.path_, ctx.blacklist_);
			subctx.path_.push_back(PathNode{i, opcode});
			lookfor(subctx, static_cast<const search::OpTrieT::NodeT*>(
				tri->next(PathNode{i, opcode})), args[i]);
			ctx.blacklist_ = subctx.blacklist_;
			ctx.symbs_ = subctx.symbs_;
			// final result is an intersect of all subresults
			for (auto it = ctx.results_.begin(), et = ctx.results_.end();
				it != et;)
			{
				if (false == estd::has(subctx.results_, it->first))
				{
					it = ctx.results_.erase(it);
				}
				else
				{
					it->second.merge(subctx.results_[it->first]);
					++it;
				}
			}
		}
	}

	void any_condition (SearchContext& ctx, const std::string& symb,
		const search::OpTrieT::NodeT* tri)
	{
		StatsMapT subresults;
		size_t tri_depth = ctx.path_.size();
		search::possible_paths(
			[&subresults, tri_depth](const search::PathListT& path, const search::PathVal& val)
			{
				size_t depth = tri_depth + path.size();
				Stats stats{depth, depth};
				for (const auto& lpair : val.leaves_)
				{
					merge_stats(subresults, lpair.second, stats);
				}
				for (const auto& apair : val.attrs_)
				{
					merge_stats(subresults, apair.second, stats);
				}
			}, tri);
		// for each subresult travel through path
		if (estd::has(selections_, symb))
		{
			for (auto& subresult : subresults)
			{
				teq::iTensor* tri_rep = search::walk(subresult.first, ctx.path_);
				// associate tri_rep with symb
				auto& symb_map = ctx.symbs_[subresult.first];
				if (estd::has(symb_map, symb) && symb_map[symb] != tri_rep)
				{
					ctx.blacklist_.emplace(subresult.first);
				}
				symb_map.emplace(symb, tri_rep);
			}
		}
		if (ctx.results_.empty())
		{
			ctx.results_ = subresults;
		}
		else
		{
			for (auto it = ctx.results_.begin(), et = ctx.results_.end();
				it != et;)
			{
				if (false == estd::has(subresults, it->first))
				{
					it = ctx.results_.erase(it);
				}
				else
				{
					it->second.merge(subresults[it->first]);
					++it;
				}
			}
		}
	}

	bool equals (const Attribute& pba, const marsh::iObject* attr)
	{
		bool match = false;
		switch (pba.attr_case())
		{
			case Attribute::kInum:
				if (auto num = dynamic_cast<const marsh::iNumber*>(attr))
				{
					match = pba.inum() == num->to_int64();
				}
				break;
			case Attribute::kDnum:
				if (auto num = dynamic_cast<const marsh::iNumber*>(attr))
				{
					match = pba.dnum() == num->to_float64();
				}
				break;
			case Attribute::kIarr:
				if (auto narr = dynamic_cast<const marsh::iArray*>(attr))
				{
					const auto& arr = pba.iarr().values();
					if ((size_t) arr.size() == narr->size())
					{
						match = true;
						narr->foreach(
						[&](size_t i, const marsh::ObjptrT& obj)
						{
							auto num = dynamic_cast<const marsh::iNumber*>(obj.get());
							match = match &&
								nullptr != num && arr[i] == num->to_int64();
						});
					}
				}
				break;
			case Attribute::kDarr:
				if (auto narr = dynamic_cast<const marsh::iArray*>(attr))
				{
					const auto& arr = pba.darr().values();
					if ((size_t) arr.size() == narr->size())
					{
						match = true;
						narr->foreach(
						[&](size_t i, const marsh::ObjptrT& obj)
						{
							auto num = dynamic_cast<const marsh::iNumber*>(obj.get());
							match = match &&
								nullptr != num && arr[i] == num->to_float64();
						});
					}
				}
				break;
			case Attribute::kStr:
				match = pba.str() == attr->to_string();
				break;
			case Attribute::kNode:
			{
				if (auto tens = dynamic_cast<const teq::TensorObj*>(attr))
				{
					SearchContext ctx;
					lookfor(ctx, from_->root(), pba.node());
					match = estd::has(ctx.results_, tens->get_tensor().get());
				}
			}
				break;
			case Attribute::kLayer:
			{
				if (auto lay = dynamic_cast<const teq::LayerObj*>(attr))
				{
					const Layer& layer = pba.layer();
					match = Layer::kName == layer.nullable_name_case() || layer.name() == lay->get_opname();
					if (match && layer.has_input())
					{
						SearchContext ctx;
						lookfor(ctx, from_->root(), layer.input());
						match = estd::has(ctx.results_, lay->get_tensor().get());
					}
				}
			}
				break;
			default:
				teq::fatal("cannot compare unknown attribute");
		}
		return match;
	}

	search::OpTrieT* from_;

	std::unordered_set<std::string> selections_;

	std::unordered_set<ConditionT> conditions_;
};

}

#endif // QUERY_HPP
