#include "query/query.hpp"

#include "tenncor/distrib/reference.hpp"

#ifdef DISTRIB_IREFERENCE_HPP

namespace distr
{

DRefptrSetT reachable_refs (teq::TensptrT root)
{
	std::stringstream inss;
	inss << "{\"op\":{\"opname\":\"" << refname << "\"}}";
	query::Node cond;
	query::json_parse(cond, inss);

	query::Query q;
	root->accept(q);
	auto owners = teq::track_owners({root});
	auto results = q.match(cond);
	DRefptrSetT refs;
	refs.reserve(results.size());
	for (auto res : results)
	{
		refs.emplace(std::static_pointer_cast<iDistRef>(
			owners[res.root_].lock()));
	}
	return refs;
}

}

#endif
