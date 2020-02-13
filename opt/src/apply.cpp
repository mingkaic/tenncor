#include "opt/apply.hpp"

#ifdef OPT_APPLY_HPP

namespace opt
{

bool optimize (GraphInfo& graph, const OptRulesT& rules)
{
	bool converted = false;
	for (const OptRule& rule : rules)
	{
		query::QResultsT results;
		query::Query q(graph.sindex_);
		rule.matcher_(q);
		q.exec(results);
		if (results.size() > 0)
		{
			converted = true;
			teq::TensMapT<teq::TensptrT> converts;
			for (query::QueryResult& result : results)
			{
				converts.emplace(result.root_,
					rule.target_->convert(result.symbs_));
			}
			graph.replace(converts);
		}
	}
	return converted;
}

}

#endif
