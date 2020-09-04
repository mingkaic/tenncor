#include "internal/teq/traveler.hpp"

#ifdef TEQ_TRAVELER_HPP

namespace teq
{

TensptrsT get_alldeps (iFunctor& func)
{
	auto deps = func.get_argndeps();
	auto attrs = func.ls_attrs();
	TensptrsT out = deps;
	out.reserve(deps.size() + attrs.size());
	for (auto attr : attrs)
	{
		if (auto tens_attr = dynamic_cast<const TensorRef*>(
			func.get_attr(attr)))
		{
			out.push_back(tens_attr->get_tensor());
		}
	}
	return out;
}

OwnMapT convert_ownmap (const RefMapT& refs)
{
	OwnMapT owners;
	for (const auto& rpair : refs)
	{
		if (false == rpair.second.expired())
		{
			owners.emplace(rpair.first, rpair.second.lock());
		}
	}
	return owners;
}

}

#endif
