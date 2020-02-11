
#ifndef EXPERIMENTAL_OPT_RULE_HPP
#define EXPERIMENTAL_OPT_RULE_HPP

#include "query/query.hpp"

#include "experimental/opt/target.hpp"

namespace opt
{

using MatcherF = std::function<void(query::Query& q)>;

struct OptRule final
{
	MatcherF matcher_;

	TargptrT target_;
};

using OptRulesT = std::vector<OptRule>;

}

#endif // EXPERIMENTAL_OPT_RULE_HPP
