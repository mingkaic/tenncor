//
/// query.hpp
/// query
///
/// Purpose:
/// Define subgraph filtering functionality
///

#ifndef QUERY_QUERY_HPP
#define QUERY_QUERY_HPP

#include <optional>

#include "query/sindex.hpp"
#include "query/position.hpp"
#include "query/parse.hpp"

namespace query
{

using ConditionT = std::shared_ptr<Node>;

static inline teq::Shape to_shape2 (
	const google::protobuf::RepeatedField<uint32_t>& sfields)
{
	std::vector<teq::DimT> slist(sfields.begin(), sfields.end());
	return teq::Shape(slist);
}

inline bool equals2 (double scalar, const teq::iLeaf* leaf)
{
	return teq::IMMUTABLE == leaf->get_usage() &&
		leaf->to_string() == fmts::to_string(scalar);
}

inline bool equals2 (const Leaf& var, const teq::iLeaf* leaf)
{
	return
		(Leaf::kLabel != var.nullable_label_case() ||
			var.label() == leaf->to_string()) &&
		(Leaf::kDtype != var.nullable_dtype_case() ||
			var.dtype() == leaf->type_label()) &&
		(Leaf::kUsage != var.nullable_usage_case() ||
			var.usage() == teq::get_usage_name(leaf->get_usage())) &&
		(0 == var.shape_size() ||
			to_shape2(var.shape()).compatible_after(leaf->shape(), 0));
}

inline bool equals2 (const Attribute& pba, const marsh::iObject* attr,
	std::function<void(teq::TensSetT&,const Node&)> find_node)
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
					[&](size_t i, const marsh::iObject* obj)
					{
						auto num = dynamic_cast<const marsh::iNumber*>(obj);
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
					[&](size_t i, const marsh::iObject* obj)
					{
						auto num = dynamic_cast<const marsh::iNumber*>(obj);
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
				teq::TensSetT candidates;
				find_node(candidates, pba.node());
				match = estd::has(candidates, tens->get_tensor().get());
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
					teq::TensSetT candidates;
					find_node(candidates, layer.input());
					match = estd::has(candidates, lay->get_tensor().get());
				}
			}
		}
			break;
		default:
			teq::fatal("cannot compare unknown attribute");
	}
	return match;
}

using SymbMapT = std::unordered_map<std::string,teq::iTensor*>;

struct QueryResult final
{
	teq::iTensor* root_;

	SymbMapT symbs_;
};

using QResultsT = std::vector<QueryResult>;

using Paths2T = std::vector<PathNodesT>;

inline teq::iTensor* walk (teq::iTensor* root, const PathNodesT& path)
{
	for (const PathNode& node : path)
	{
		auto f = dynamic_cast<teq::iFunctor*>(root);
		if (nullptr == f)
		{
			return nullptr;
		}
		auto children = f->get_children();
		if (node.idx_ >= children.size())
		{
			return nullptr;
		}
		root = children.at(node.idx_).get();
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
			const Paths2T& paths = pathpair.second;
			if (paths.empty())
			{
				return std::optional<SymbMapT>();
			}
			teq::iTensor* srep = walk(src, paths.front());
			for (auto it = paths.begin() + 1, et = paths.end(); it != et; ++it)
			{
				if (srep != walk(src, *it))
				{
					return std::optional<SymbMapT>();
				}
			}
			// associate srep with symb
			out.emplace(symb, srep);
		}
		return out;
	}

	PosMapT results_;

	std::unordered_set<std::string> selections_;

	std::unordered_map<std::string,Paths2T> captures_;
};

// Information changing with each iteration
struct SearchIterator final
{
	SearchIterator (const search::OpTrieT::NodeT* trinode) :
		pathnode_(PathNode{0, ""}),
		trinode_(trinode), consume_(union_position) {}

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

	void consume (Transaction& ctx, const PosMapT& stats) const
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
	auto& leaves = path.last_trie()->leaf_->leaves_;
	teq::LeafsT leafkeys = estd::get_keys(leaves);
	std::sort(leafkeys.begin(), leafkeys.end(),
		[](teq::iLeaf* a, teq::iLeaf* b)
		{
			return a->to_string() < b->to_string();
		});
	for (teq::iLeaf* leaf : leafkeys)
	{
		if (equals2(cond, leaf))
		{
			nomatch = false;
			PosMapT smap;
			bind_position(smap, leaves.at(leaf), leaf, path.size());
			path.consume(ctx, smap);
		}
	}
	if (nomatch)
	{
		ctx.fail();
	}
}

// Return true if there's at least one result
void lookfor (Transaction& ctx, const SearchList& path,
	const Node& cond);

struct Query
{
	Query (const search::OpTrieT& sindex) : from_(&sindex) {}

	// Return query with <existing symbol(s)> AND <symb>
	Query select (const std::string& symb) const
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
	Query where (const ConditionT& condition) const
	{
		auto new_conditions = conditions_;
		new_conditions.emplace(condition);
		return Query(
			from_,
			selections_,
			new_conditions);
	}

	Query where (std::istream& condition) const
	{
		auto cond = std::make_shared<Node>();
		json_parse(*cond, condition);
		return where(cond);
	}

	// set depth_asc to true if max-distance from condition roots
	// to leaf/symbol and sum paths is sorted by min-max
	void exec (QResultsT& results, bool depth_asc = true) const
	{
		PosMapT order;
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
				std::string astr = a.root_->to_string() + ":" +
					a.root_->shape().to_string();
				std::string bstr = b.root_->to_string() + ":" +
					b.root_->shape().to_string();
				if (astr == bstr)
				{
					if (depth_asc)
					{
						return order[a.root_] < order[b.root_];
					}
					return order[a.root_] > order[b.root_];
				}
				if (depth_asc)
				{
					return astr < bstr;
				}
				return astr > bstr;
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

#endif // QUERY_QUERY_HPP
