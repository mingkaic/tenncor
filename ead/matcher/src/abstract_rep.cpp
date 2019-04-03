#include "ead/matcher/abstract_rep.hpp"

#ifdef EAD_ABSTRACT_REP_HPP

namespace ead
{

GraphRep represent (ade::TensT tensors)
{
	ArgRepsT roots;
	roots.reserve(tensors.size());
	Abstracizer abstracizer;
	for (ade::TensptrT& tens : tensors)
	{
		tens->accept(abstracizer);
		roots.push_back(abstracizer.get_arg(tens));
	}
	return GraphRep{roots, abstracizer.leaf_map_};
}

void optimize (GraphRep& graph)
{
    // todo: implement
}

}

#endif
