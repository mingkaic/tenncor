#include "internal/teq/traveler.hpp"

#ifdef TEQ_TRAVELER_HPP

namespace teq
{

TensptrsT get_deps (iFunctor& func)
{
	return func.get_dependencies();
}

TensptrsT get_args (iFunctor& func)
{
	return func.get_args();
}

TensptrsT get_attrs (iFunctor& func)
{
	auto attrs = func.ls_attrs();
	TensptrsT out;
	out.reserve(attrs.size());
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
		owners.emplace(rpair.first, rpair.second.lock());
	}
	return owners;
}

}

#endif
