
#include "distrib/reference.hpp"

#ifdef DISTRIB_REFERENCE_HPP

namespace distr
{

void separate_by_server (
	estd::StrMapT<estd::StrSetT>& out,
	const DRefSetT& refs)
{
	for (auto ref : refs)
	{
		out[ref->cluster_id()].emplace(ref->node_id());
	}
}

}

#endif
