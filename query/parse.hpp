//
/// parse.hpp
/// query
///
/// Purpose:
/// Define subgraph filtering condition parsing
///

#ifndef QUERY_PARSE_HPP
#define QUERY_PARSE_HPP

#include "teq/objs.hpp"
#include "teq/ileaf.hpp"

#include "query/query.pb.h"

namespace query
{

using ConditionT = std::shared_ptr<Node>;

static inline teq::Shape to_shape (
	const google::protobuf::RepeatedField<uint32_t>& sfields)
{
	std::vector<teq::DimT> slist(sfields.begin(), sfields.end());
	return teq::Shape(slist);
}

inline bool equals (double scalar, const teq::iLeaf* leaf)
{
	return teq::IMMUTABLE == leaf->get_usage() &&
		leaf->to_string() == fmts::to_string(scalar);
}

inline bool equals (const Leaf& var, const teq::iLeaf* leaf)
{
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

inline bool equals (const Attribute& pba, const marsh::iObject* attr,
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

void json_parse (Node& condition, std::istream& json_in);

}

#endif // QUERY_PARSE_HPP
