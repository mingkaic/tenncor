#include "experimental/opt/apply.hpp"

#ifdef EXPERIMENTAL_OPT_APPLY_HPP

namespace opt
{

void optimize (GraphInfo& graph, const OptRulesT& rules)
{
	for (const OptRule& rule : rules)
	{
		query::QResultsT results;
		query::Query q(graph.sindex_);
		rule.matcher_(q);
		q.exec(results);
		if (results.size() > 0)
		{
			teq::TensMapT<teq::TensptrT> converts;
			for (query::QueryResult& result : results)
			{
				converts.emplace(result.root_,
					rule.target_->convert(result.symbs_));
			}
			graph.replace(converts);
		}
	}
}

}

#endif
