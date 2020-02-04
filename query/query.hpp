//
/// query.hpp
/// query
///
/// Purpose:
/// Define subgraph filtering functionality
///

#ifndef QUERY_HPP
#define QUERY_HPP

#include <optional>

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

using PathsT = std::vector<PathNodesT>;

inline teq::iTensor* walk (teq::iTensor* root, const PathNodesT& path)
{
	for (const PathNode& node : path)
	{
		root = static_cast<teq::iFunctor*>(root)->
			get_children().at(node.idx_).get();
	}
	return root;
}

// Information kept throughout searching
struct Transaction final
{
	Transaction (void) = default;

	void fail (void)
	{
		results_.clear();
	}

	// return valid symbol map, if paths from src to symbols do not conflict
	std::optional<SymbMapT> selection (teq::iTensor* src) const
	{
		SymbMapT out;
		for (auto& pathpair : captures_)
		{
			std::string symb = pathpair.first;
			const PathsT& paths = pathpair.second;
			if (paths.empty())
			{
				return std::optional<SymbMapT>();
			}
			teq::iTensor* srep = walk(src, paths.front());
			for (auto it = paths.begin() + 1, et = paths.end(); it != et; ++it)
			{
				if (srep != walk(src, *it))
				{
					teq::warnf("symbol %s has conflicting matches",
						symb.c_str());
				}
			}
			// associate srep with symb
			out.emplace(symb, srep);
		}
		return out;
	}

	StatsMapT results_;

	std::unordered_set<std::string> selections_;

	std::unordered_map<std::string,PathsT> captures_;
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

	SearchIterator* next (const PathNode& node)
	{
		next_ = std::make_shared<SearchIterator>(node,
			static_cast<const search::OpTrieT::NodeT*>(
				trinode_->next(node)), consume_);
		return next_.get();
	}

	SearchIterator* next (const PathNode& node, TxConsumeF consume)
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

	void consume (Transaction& ctx, const StatsMapT& stats) const
	{
		last_->consume_(ctx.results_, stats);
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
			path.consume(ctx, smap);
		}
	}
	if (nomatch)
	{
		ctx.fail();
	}
}

PathNodesT pathify (const SearchList& list)
{
	PathNodesT path;
	path.reserve(list.size());
	for (auto it = list.begin_, et = list.last_->next_.get();
		it != et; it = it->next_.get())
	{
		if (it->pathnode_.op_ != egen::BAD_OP)
		{
			path.push_back(it->pathnode_);
		}
	}
	return path;
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
		ctx.captures_[symb].push_back(pathify(path));
	}
	path.consume(ctx, subresults);
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
	StatsMapT results;
	lookfor(ctx, path.next(PathNode{0, opcode},
		[&results](StatsMapT& out, const StatsMapT& other)
		{
			results = other;
		}), args[0]);
	for (size_t i = 1, n = args.size();
		i < n && false == results.empty(); ++i)
	{
		StatsMapT subresults;
		lookfor(ctx, path.next(PathNode{i, opcode},
			[&subresults](StatsMapT& out, const StatsMapT& other)
			{
				subresults = other;
			}), args[i]);
		intersect_stats(results, subresults);
	}
	path.consume(ctx, results);
}

void match_attrs (search::FSetMapT& out,
	const google::protobuf::Map<std::string,Attribute>& targets,
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
	const Node& cond)
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
				StatsMapT attr_results;
				TxConsumeF orig_consume = path.last_->consume_;
				path.last_trie() = path.front_trie();
				path.last_->consume_ =
				[&attr_results, &attr_matches](StatsMapT& out, const StatsMapT& other)
				{
					for (const auto& apair : other)
					{
						auto f = dynamic_cast<teq::iFunctor*>(apair.first);
						if (estd::has(attr_matches, f))
						{
							for (auto target_root : attr_matches.at(f))
							{
								attr_results.emplace(target_root, apair.second);
							}
						}
					}
				};
				iterate_condition(ctx, path, op);
				if (attr_results.empty())
				{
					ctx.fail();
				}
				else
				{
					orig_consume(ctx.results_, attr_results);
				}
			}
			else
			{
				iterate_condition(ctx, path, op);
			}
			if (Operator::kCapture == op.nullable_capture_case() &&
				estd::has(ctx.selections_, op.capture()))
			{
				ctx.captures_[op.capture()].push_back(pathify(path));
			}
		}
			break;
		default:
			teq::fatal("cannot look for unknown node");
	}
}

struct Query
{
	Query (const search::OpTrieT& sindex) : from_(&sindex) {}

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
				else if (std::optional<SymbMapT> symbs =
					ctx.selection(batch_pair.first))
				{
					results.push_back(QueryResult{
						batch_pair.first, *symbs});
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
	Query (const search::OpTrieT* from,
		const std::unordered_set<std::string>& selections,
		const std::unordered_set<ConditionT>& conditions) :
		from_(from), selections_(selections), conditions_(conditions) {}

	const search::OpTrieT* from_;

	std::unordered_set<std::string> selections_;

	std::unordered_set<ConditionT> conditions_;
};

}

#endif // QUERY_HPP
