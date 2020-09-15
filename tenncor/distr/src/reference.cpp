
#include "tenncor/distr/reference.hpp"

#ifdef DISTR_REFERENCE_HPP

namespace distr
{

void separate_by_server (
	types::StrUMapT<types::StrUSetT>& out,
	const DRefSetT& refs)
{
	for (auto ref : refs)
	{
		out[ref->cluster_id()].emplace(ref->node_id());
	}
}

}

#endif
