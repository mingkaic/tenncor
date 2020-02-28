#include "opt/graph.hpp"

#ifdef OPT_GRAPH_HPP

namespace opt
{

OwnersT convert_ownermap (const teq::OwnerMapT& omap)
{
	OwnersT owners;
	for (const auto& opair : omap)
	{
		owners.emplace(opair.first, opair.second.lock());
	}
	return owners;
}

}

#endif
