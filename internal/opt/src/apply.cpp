#include "internal/opt/apply.hpp"

#ifdef OPT_APPLY_HPP

namespace opt
{

bool optimize (GraphInfo& graph, const OptRulesT& rules)
{
	bool converted = false;
	for (const OptRule& rule : rules)
	{
		query::QResultsT results;
		for (auto& match_src : rule.match_srcs_)
		{
			auto res = graph.sindex_.match(match_src);
			results.insert(results.end(), res.begin(), res.end());
		}
		if (results.size() > 0)
		{
			converted = true;
			teq::OwnMapT converts;
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
