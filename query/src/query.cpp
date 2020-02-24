
#include "query/query.hpp"

#ifdef QUERY_QUERY_HPP

namespace query
{

bool equals (const marsh::iObject* attr,
	const query::Attribute& pba, const Query& matcher)
{
	bool match = false;
	switch (pba.attr_case())
	{
		case query::Attribute::kInum:
			if (auto num = dynamic_cast<const marsh::iNumber*>(attr))
			{
				match = pba.inum() == num->to_int64();
			}
			break;
		case query::Attribute::kDnum:
			if (auto num = dynamic_cast<const marsh::iNumber*>(attr))
			{
				match = pba.dnum() == num->to_float64();
			}
			break;
		case query::Attribute::kIarr:
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
		case query::Attribute::kDarr:
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
		case query::Attribute::kStr:
			if (nullptr != attr)
			{
				match = pba.str() == attr->to_string();
			}
			break;
		case query::Attribute::kNode:
			if (auto tens = dynamic_cast<const teq::TensorObj*>(attr))
			{
				auto candidates = matcher.match(pba.node());
				teq::TensSetT candset(candidates.begin(), candidates.end());
				match = estd::has(candset, tens->get_tensor().get());
			}
			break;
		case query::Attribute::kLayer:
			if (auto lay = dynamic_cast<const teq::LayerObj*>(attr))
			{
				const Layer& layer = pba.layer();
				match = Layer::kName == layer.nullable_name_case() || layer.name() == lay->get_opname();
				if (match && layer.has_input())
				{
					auto candidates = matcher.match(layer.input());
					teq::TensSetT candset(candidates.begin(), candidates.end());
					match = estd::has(candset, lay->get_tensor().get());
				}
			}
			break;
		default:
			teq::fatal("cannot compare unknown attribute");
	}
	return match;
}

}

#endif
