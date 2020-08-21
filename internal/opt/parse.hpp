
#ifndef OPT_PARSE_HPP
#define OPT_PARSE_HPP

#include "internal/opt/rule.hpp"
#include "internal/opt/graph.hpp"

namespace opt
{

void json_parse (OptRulesT& rules,
	std::istream& json_in, const iTargetFactory& tfactory);

marsh::iObject* parse (const query::Attribute& pba, const opt::GraphInfo& graphinfo);

}

#endif // OPT_PARSE_HPP
