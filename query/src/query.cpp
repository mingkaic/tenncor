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
				auto attr = apair.second;
				for (teq::iTensor* root : attr.roots_)
				{
					for (auto& spair : attr.stats_.depths_)
					{
						spair.second += depth;
					}
					smap.emplace(root, attr.stats_);
				}
				union_stats(subresults, smap);
			}
			for (const auto& cpair : val.comms_)
			{
				StatsMapT smap;
				auto comm = cpair.second;
				for (teq::iTensor* root : comm.roots_)
				{
					for (auto& spair : comm.stats_.depths_)
					{
						spair.second += depth;
					}
					smap.emplace(root, comm.stats_);
				}
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

static void comms_matches (Transaction& ctx,
	const SearchList& path, const Operator& op)
{
	egen::_GENERATED_OPCODE opcode = egen::get_op(op.opname());
	auto tri = path.last_trie();
	if (false == tri->leaf_.has_value() || tri->leaf_->comms_.empty())
	{
		ctx.fail();
		return;
	}
	const auto& args = op.args();
	const search::FSetMapT& comms = tri->leaf_->comms_;
	teq::FuncSetT comset;
	for (const auto& comm : comms)
	{
		teq::iFunctor* cf = comm.first;
		if (cf->to_string() == op.opname())
		{
			comset.emplace(cf);
		}
	}
	if (args.empty())
	{
		StatsMapT subresult;
		for (auto& cf : comset)
		{
			StatsMapT smap;
			auto& comm = comms.at(cf);
			for (teq::iTensor* root : comm.roots_)
			{
				smap.emplace(root, comm.stats_);
			}
			union_stats(subresult, smap);
		}
		path.consume(ctx, subresult);
		return;
	}
	StatsMapT subresults;
	teq::TensMapT<std::unordered_set<size_t>> indices;
	for (size_t i = 0, n = args.size(); i < n; ++i)
	{
		const Node& cond = args[i];
		auto nextpath = path.next(PathNode{i, opcode});
		nextpath.last_trie() = nextpath.front_trie();
		nextpath.last_->consume_ =
		[&](StatsMapT& out, const StatsMapT& other)
		{
			for (const auto& o : other)
			{
				subresults.emplace(o);
				indices[o.first].emplace(i);
			}
		};
		lookfor(ctx, nextpath, cond);
	}

	StatsMapT results;
	for (auto& cf : comset)
	{
		auto children = cf->get_children();
		std::vector<size_t> matches(args.size(), 0);
		for (teq::TensptrT child : children)
		{
			if (estd::has(indices, child.get()))
			{
				for (size_t index : indices[child.get()])
				{
					++matches[index];
				}
			}
		}
		// # of matches >= # needed matches -> take cf
		if (std::all_of(matches.begin(), matches.end(),
			[](size_t i) { return i > 0; }))
		{
			Stats stats;
			for (teq::TensptrT child : children)
			{
				stats.merge(subresults.at(child.get()));
			}
			for (teq::iTensor* root : comms.at(cf).roots_)
			{
				results.emplace(root, stats);
			}
		}
	}
	if (results.empty())
	{
		ctx.fail();
	}
	else
	{
		path.consume(ctx, results);
	}
}

// Return true if there's at least one result
static void iterate_condition (Transaction& ctx,
	const SearchList& path, const Operator& op)
{
	egen::_GENERATED_OPCODE opcode = egen::get_op(op.opname());
	if (egen::is_commutative(opcode))
	{
		// trie can't handle commutative operators, so handle manually
		comms_matches(ctx, path, op);
		return;
	}
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
							for (auto target_root : attr_matches.at(f).roots_)
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
