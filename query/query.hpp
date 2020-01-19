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

#include "query/parse.hpp"

namespace query
{

static inline teq::Shape to_shape (
	const google::protobuf::RepeatedField<uint32_t>& sfields)
{
	std::vector<teq::DimT> slist(sfields.begin(), sfields.end());
	return teq::Shape(slist);
}

static inline bool equals (double scalar, const teq::iLeaf* leaf)
{
	return teq::IMMUTABLE == leaf->get_usage() &&
		leaf->to_string() == fmts::to_string(scalar);
}

static inline bool equals (const Variable& var, const teq::iLeaf* leaf)
{
	return (Variable::kLabel != var.nullable_label_case() || var.label() == leaf->to_string()) &&
		(Variable::kDtype != var.nullable_dtype_case() || var.dtype() == leaf->type_label()) &&
		(0 == var.shape_size() || to_shape(var.shape()).compatible_after(leaf->shape(), 0));
}

struct Query
{
	Query (search::OpTrieT& sindex) : sindex_(sindex) {}

	void where (teq::TensSetT& results, std::istream& condition)
	{
		Node cond;
		json_parse(cond, condition);
		constraint(results, sindex_.root(), cond);
	}

private:
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
					if (arr.size() == narr->size())
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
					if (arr.size() == narr->size())
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
					teq::TensSetT results;
					constraint(results, sindex_.root(), pba.node());
					match = estd::has(results, tens->get_tensor().get());
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
						teq::TensSetT results;
						constraint(results, sindex_.root(), layer.input());
						match = estd::has(results, lay->get_tensor().get());
					}
				}
			}
				break;
			default:
				logs::fatal("cannot compare unknown attribute");
		}
		return match;
	}

	// Return true if there's at least one result
	void constraint (teq::TensSetT& results,
		const search::OpTrieT::NodeT* tri, const Node& cond) // todo: add filter_out mechanism instead of allocing a subresult at every branch
	{
		if (nullptr == tri)
		{
			results.clear();
			return;
		}
		switch (cond.val_case())
		{
			case Node::ValCase::kCst:
			{
				if (false == tri->leaf_.has_value())
				{
					results.clear();
					return;
				}
				bool nomatch = true;
				double scalar = cond.cst();
				for (const auto& lpair : tri->leaf_->leaves_)
				{
					if (::query::equals(scalar, lpair.first))
					{
						nomatch = false;
						results.insert(lpair.second.begin(), lpair.second.end());
					}
				}
				if (nomatch)
				{
					results.clear();
				}
			}
				break;
			case Node::ValCase::kVar:
			{
				if (false == tri->leaf_.has_value())
				{
					results.clear();
					return;
				}
				bool nomatch = true;
				const Variable& var = cond.var();
				for (const auto& lpair : tri->leaf_->leaves_)
				{
					if (::query::equals(var, lpair.first))
					{
						nomatch = false;
						results.insert(lpair.second.begin(), lpair.second.end());
					}
				}
				if (nomatch)
				{
					results.clear();
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
						results.clear();
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
					teq::TensSetT subresults;
					iterate_condition(subresults, sindex_.root(), op);
					// get attr_matches[subresults intersection attr_matches.keys]
					if (subresults.empty())
					{
						results.clear();
					}
					for (const auto& apair : attr_matches)
					{
						if (estd::has(subresults, apair.first))
						{
							results.insert(apair.second.begin(), apair.second.end());
						}
					}
				}
				else
				{
					iterate_condition(results, tri, op);
				}
			}
				break;
			default:
				search::possible_paths(
					[&results](const search::PathListT& path, const search::PathVal& val)
					{
						for (const auto& lpair : val.leaves_)
						{
							results.insert(lpair.second.begin(), lpair.second.end());
						}
						for (const auto& apair : val.attrs_)
						{
							results.insert(apair.second.begin(), apair.second.end());
						}
					}, tri);
		}
	}

	// Return true if there's at least one result
	void iterate_condition (teq::TensSetT& results,
		const search::OpTrieT::NodeT* tri, const Operator& op)
	{
		egen::_GENERATED_OPCODE opcode = egen::get_op(op.opname());
		const auto& args = op.args();
		if (args.empty())
		{
			constraint(results, tri, Node());
			return;
		}
		constraint(results, static_cast<const search::OpTrieT::NodeT*>(
			tri->next(PathNode{0, opcode})), args[0]);
		teq::TensSetT subresults;
		for (size_t i = 1, n = args.size(); i < n && false == results.empty(); ++i)
		{
			constraint(subresults, static_cast<const search::OpTrieT::NodeT*>(
				tri->next(PathNode{i, opcode})), args[i]);
			// final result is an intersect of all subresults
			for (auto it = results.begin(), et = results.end();
				it != et;)
			{
				if (false == estd::has(subresults, *it))
				{
					it = results.erase(it);
				}
				else
				{
					++it;
				}
			}
			subresults.clear();
		}
	}

	search::OpTrieT& sindex_;
};

}

#endif // QUERY_HPP
