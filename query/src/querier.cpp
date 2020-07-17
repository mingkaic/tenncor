
#include "query/querier.hpp"

#ifdef QUERY_QUERIER_HPP

namespace query
{

inline bool doub_eq (const double& a, const double& b)
{
	return std::fabs(a - b) < std::numeric_limits<float>::epsilon();
}

bool equals (
	QResultsT& candidates, const marsh::iObject* attr,
	const Attribute& pba, const Query& matcher)
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
				match = doub_eq(pba.dnum(), num->to_float64());
			}
			break;
		case Attribute::kIarr:
			if (auto narr = dynamic_cast<const marsh::iArray*>(attr))
			{
				const auto& arr = pba.iarr().values();
				if ((size_t) arr.size() == narr->size() && narr->is_integral())
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
				if ((size_t) arr.size() == narr->size() && false == narr->is_integral())
				{
					match = true;
					narr->foreach(
					[&](size_t i, const marsh::iObject* obj)
					{
						auto num = dynamic_cast<const marsh::iNumber*>(obj);
						match = match &&
							nullptr != num && doub_eq(arr[i], num->to_float64());
					});
				}
			}
			break;
		case Attribute::kStr:
			if (nullptr != attr)
			{
				match = pba.str() == attr->to_string();
			}
			break;
		case Attribute::kNode:
			if (auto tens = dynamic_cast<const teq::TensorObj*>(attr))
			{
				candidates = matcher.match(pba.node());
				estd::remove_if(candidates,
					[&tens](const QueryResult& result)
					{
						return result.root_ != tens->get_tensor().get();
					});
				match = candidates.size() > 0;
			}
			break;
		case Attribute::kLayer:
			if (auto lay = dynamic_cast<const teq::LayerObj*>(attr))
			{
				const Layer& layer = pba.layer();
				match = Layer::kNameNil == layer.nullable_name_case() || layer.name() == lay->get_opname();
				if (match && layer.has_input())
				{
					candidates = matcher.match(layer.input());
					estd::remove_if(candidates,
						[&lay](const QueryResult& result)
						{
							return result.root_ != lay->get_tensor().get();
						});
					match = candidates.size() > 0;
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
