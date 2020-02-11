#include "experimental/opt/graph.hpp"

#ifdef EXPERIMENTAL_OPT_GRAPH_HPP

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
