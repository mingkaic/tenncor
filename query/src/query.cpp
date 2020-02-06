#include "query/query.hpp"

#ifdef QUERY_QUERY_HPP

namespace query
{

static PathNodesT pathify (const SearchList& list)
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

static void any_condition (Transaction& ctx,
	const SearchList& path, const std::string& symb = "")
{
	StatsMapT subresults;
	size_t tri_depth = path.size();
	search::possible_paths(
		[&subresults, tri_depth](
			const search::PathListT& path, const search::PathVal& val)
		{
			size_t depth = tri_depth + path.size();
			for (const auto& lpair : val.leaves_)
			{
				StatsMapT smap;
				bind_stats(smap, lpair.second, lpair.first, depth);
				union_stats(subresults, smap);
			}
			for (const auto& apair : val.attrs_)
			{
				StatsMapT smap;
				bind_stats(smap, apair.second, apair.first, depth);
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

// Return true if there's at least one result
static void iterate_condition (Transaction& ctx,
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
			union_stats(results, other);
		}), args[0]);
	for (size_t i = 1, n = args.size();
		i < n && false == results.empty(); ++i)
	{
		StatsMapT subresults;
		lookfor(ctx, path.next(PathNode{i, opcode},
			[&subresults](StatsMapT& out, const StatsMapT& other)
			{
				union_stats(subresults, other);
			}), args[i]);
		intersect_stats(results, subresults);
	}
	path.consume(ctx, results);
}

static void match_attrs (search::FSetMapT& out,
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
		case Node::ValCase::kLeaf:
			match_condition(ctx, path, cond.leaf());
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

}

#endif
