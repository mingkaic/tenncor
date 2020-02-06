
#ifndef EXPERIMENTAL_OPT_RULE_HPP
#define EXPERIMENTAL_OPT_RULE_HPP

#include "query/query.hpp"

namespace opt
{

using MatcherF = std::function<void(query::Query& q)>;

struct OptRule final
{
	MatcherF matcher_;

	TargptrT target_;
};

using OptRulesT = std::vector<OptRule>;

struct GraphInfo final
{
	GraphInfo (const query::search::OpTrieT& trie,
		const teq::OwnerMapT& owner) : query_(trie), owner_(owner) {}

	teq::TensptrsT find (const query::Node& condition) const
	{
		query::QResultsT results;
		query_.where(std::make_shared<query::Node>(condition)).exec(results);
		teq::TensptrsT outs;
		outs.reserve(results.size());
		std::transform(results.begin(), results.end(),
			std::back_inserter(outs),
			[this](const query::QueryResult& result)
			{
				return owner_.at(result.root_).lock();
			});
		return outs;
	}

	query::Query query_;

	teq::OwnerMapT owner_;
};

}

#endif // EXPERIMENTAL_OPT_RULE_HPP
