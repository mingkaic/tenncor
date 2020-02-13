
#ifndef OPT_RULE_HPP
#define OPT_RULE_HPP

#include "query/query.hpp"

#include "opt/target.hpp"

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

#endif // OPT_RULE_HPP
