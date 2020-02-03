//
/// query.hpp
/// query
///
/// Purpose:
/// Define subgraph filtering functionality
///

#ifndef QUERY_HPP
#define QUERY_HPP

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

// Information kept throughout searching
struct Transaction final
{
	Transaction (void) = default;

	Transaction (const teq::TensSetT& blacklist) :
		blacklist_(blacklist) {}

	void fail (void)
	{
		results_.clear();
	}

	StatsMapT results_;

	teq::TensSetT blacklist_;

	teq::TensMapT<SymbMapT> symbs_;

	std::unordered_set<std::string> selections_;
};

// Information changing with each iteration
struct SearchIterator final
{
	SearchIterator (const search::OpTrieT::NodeT* trinode) :
		pathnode_(PathNode{0, egen::BAD_OP}),
		trinode_(trinode), consume_(union_stats) {}

	SearchIterator (const PathNode& path,
		const search::OpTrieT::NodeT* trinode, TxConsumeF consume) :
		pathnode_(path), trinode_(trinode), consume_(consume) {}

	SearchIterator* next (const PathNode& node,
		TxConsumeF consume = union_stats)
	{
		next_ = std::make_shared<SearchIterator>(node,
			static_cast<const search::OpTrieT::NodeT*>(
				trinode_->next(node)), consume);
		return next_.get();
	}

	PathNode pathnode_;
	const search::OpTrieT::NodeT* trinode_;
	TxConsumeF consume_;

	// iteration info
	SearchIterator* prev_ = nullptr;
	std::shared_ptr<SearchIterator> next_ = nullptr;
};

struct SearchList final
{
	SearchList (SearchIterator* begin) :
		begin_(begin), last_(begin), size_(0) {}

	SearchList (SearchIterator* begin, SearchIterator* last, size_t n) :
		begin_(begin), last_(last), size_(n) {}

	SearchList next (const PathNode& node) const
	{
		return SearchList(begin_, last_->next(node), size_ + 1);
	}

	SearchList next (const PathNode& node, TxConsumeF consume) const
	{
		return SearchList(begin_, last_->next(node, consume), size_ + 1);
	}

	const search::OpTrieT::NodeT*& front_trie (void) const
	{
		return begin_->trinode_;
	}

	const search::OpTrieT::NodeT*& last_trie (void) const
	{
		return last_->trinode_;
	}

	size_t size (void) const
	{
		return size_;
	}

	SearchIterator* begin_;

	SearchIterator* last_;

	size_t size_;
};

template <typename T>
void match_condition (Transaction& ctx, SearchList path, T cond)
{
	if (false == path.last_trie()->leaf_.has_value())
	{
		ctx.fail();
		return;
	}
	bool nomatch = true;
	Stats stats{path.size(), path.size()};
	for (const auto& lpair : path.last_trie()->leaf_->leaves_)
	{
		if (equals(cond, lpair.first))
		{
			nomatch = false;
			StatsMapT smap;
			bind_stats(smap, lpair.second, stats);
			path.last_->consume_(ctx.results_, smap);
		}
	}
	if (nomatch)
	{
		ctx.fail();
	}
}

inline teq::iTensor* walk (teq::iTensor* root, const SearchList& path)
{
	for (auto it = path.begin_, et = path.last_->next_.get();
		it != et; it = it->next_.get())
	{
		if (it->pathnode_.op_ != egen::BAD_OP)
		{
			root = static_cast<teq::iFunctor*>(root)->
				get_children().at(it->pathnode_.idx_).get();
		}
	}
	return root;
}

void any_condition (Transaction& ctx,
	const SearchList& path, const std::string& symb = "")
{
	StatsMapT subresults;
	size_t tri_depth = path.size();
	search::possible_paths(
		[&subresults, tri_depth](
			const search::PathListT& path, const search::PathVal& val)
		{
			size_t depth = tri_depth + path.size();
			Stats stats{depth, depth};
			for (const auto& lpair : val.leaves_)
			{
				StatsMapT smap;
				bind_stats(smap, lpair.second, stats);
				union_stats(subresults, smap);
			}
			for (const auto& apair : val.attrs_)
			{
				StatsMapT smap;
				bind_stats(smap, apair.second, stats);
				union_stats(subresults, smap);
			}
		}, path.last_trie());
	// for each subresult travel through path
	if (estd::has(ctx.selections_, symb))
	{
		for (auto& subresult : subresults)
		{
			teq::iTensor* tri_rep = walk(subresult.first, path);
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
		path.last_->consume_ = union_stats;
	}
	else
	{
		path.last_->consume_ = intersect_stats;
	}
	path.last_->consume_(ctx.results_, subresults);
}

void lookfor (Transaction& ctx, const SearchList& path,
	const Node& cond);

// Return true if there's at least one result
void iterate_condition (Transaction& ctx,
	const SearchList& path, const Operator& op)
{
	egen::_GENERATED_OPCODE opcode = egen::get_op(op.opname());
	const auto& args = op.args();
	if (args.empty())
	{
		any_condition(ctx, path.next(PathNode{0, opcode}));
		return;
	}
	lookfor(ctx, path.next(PathNode{0, opcode}), args[0]);
	for (size_t i = 1, n = args.size();
		i < n && false == ctx.results_.empty(); ++i)
	{
		Transaction subctx(ctx.blacklist_);
		subctx.selections_ = ctx.selections_;
		lookfor(subctx, path.next(PathNode{i, opcode}), args[i]);
		ctx.blacklist_ = subctx.blacklist_;
		ctx.symbs_ = subctx.symbs_;
		// final result is an intersect of all subresults
		intersect_stats(ctx.results_, subctx.results_);
	}
}

void match_attrs (search::FSetMapT& out,
	const google::protobuf::Map<std::string,Attribute >& targets,
	const search::OpTrieT::NodeT* trie_root,
	const search::OpTrieT::NodeT* node)
{
	if (false == node->leaf_.has_value())
	{
		return;
	}
	for (const auto& attr : node->leaf_->attrs_)
	{
		teq::iFunctor* iattr = attr.first;
		std::unordered_set<std::string> need_keys;
		// add functor if all attributes match
		if (std::all_of(targets.begin(), targets.end(),
			[&](const auto& pb_attr)
			{
				std::string attrkey = pb_attr.first;
				auto val = iattr->get_attr(attrkey);
				return nullptr != val && equals(pb_attr.second, val,
					[&](teq::TensSetT& results, const Node& cond)
					{
						Transaction ctx;
						SearchIterator begin(trie_root);
						lookfor(ctx, SearchList(&begin), cond);
						for (auto& res :ctx.results_)
						{
							results.emplace(res.first);
						}
					});
			}))
		{
			out.emplace(attr);
		}
	}
}

// Return true if there's at least one result
void lookfor (Transaction& ctx, const SearchList& path,
	const Node& cond) // todo: add filter_out mechanism instead of allocing a subresult at every branch
{
	auto& tri = path.last_trie();
	if (nullptr == tri)
	{
		ctx.fail();
		return;
	}
	switch (cond.val_case())
	{
		case Node::ValCase::kCst:
			match_condition(ctx, path, cond.cst());
			break;
		case Node::ValCase::kVar:
			match_condition(ctx, path, cond.var());
			break;
		case Node::ValCase::kSymb:
			any_condition(ctx, path, cond.symb());
			break;
		case Node::ValCase::kOp:
		{
			const Operator& op = cond.op();
			const auto& attrs = op.attrs();
			if (attrs.size() > 0)
			{
				auto lookahead = static_cast<const search::OpTrieT::NodeT*>(
					tri->next(PathNode{0, egen::get_op(op.opname())}));
				search::FSetMapT attr_matches;
				match_attrs(attr_matches, attrs, path.front_trie(), lookahead);
				if (attr_matches.empty())
				{
					ctx.fail();
					return;
				}
				// match the rest of the condition subgraph from trie root
				// in order to filter for matching attributable functors
				Transaction subctx(ctx.blacklist_);
				subctx.selections_ = ctx.selections_;
				path.last_trie() = path.front_trie();
				iterate_condition(subctx, path, op);
				ctx.blacklist_ = subctx.blacklist_;
				ctx.symbs_ = subctx.symbs_;
				// get attr_matches[subresults intersection attr_matches.keys]
				if (subctx.results_.empty())
				{
					ctx.fail();
				}
				StatsMapT subresults;
				for (const auto& apair : attr_matches)
				{
					if (estd::has(subctx.results_, apair.first))
					{
						for (auto croot : apair.second)
						{
							subresults.emplace(croot,
								subctx.results_[apair.first]);
						}
					}
				}
				union_stats(ctx.results_, subresults);
			}
			else
			{
				iterate_condition(ctx, path, op);
			}
		}
			break;
		default:
			teq::fatal("cannot look for unknown node");
	}
}

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
			Transaction ctx;
			ctx.selections_ = selections_;
			SearchIterator begin(root);
			lookfor(ctx, SearchList(&begin), *cond);
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

	search::OpTrieT* from_;

	std::unordered_set<std::string> selections_;

	std::unordered_set<ConditionT> conditions_;
};

}

#endif // QUERY_HPP
