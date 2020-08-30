#include "internal/teq/traveler.hpp"

#ifdef TEQ_TRAVELER_HPP

namespace teq
{

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
