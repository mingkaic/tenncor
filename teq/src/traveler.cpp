#include "teq/traveler.hpp"

#ifdef TEQ_TRAVELER_HPP

namespace teq
{

TensptrsT get_alldeps (iFunctor& func)
{
	auto deps = func.get_dependencies();
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

}

#endif
