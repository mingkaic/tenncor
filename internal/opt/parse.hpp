
#ifndef OPT_PARSE_HPP
#define OPT_PARSE_HPP

#include "internal/opt/optimize.pb.h"

#include "internal/opt/rule.hpp"
#include "internal/opt/graph.hpp"

namespace opt
{

void json2optimization (Optimization& pb_opt, std::istream& json_in);

void parse_optimization (OptRulesT& rules,
	const Optimization& pb_opt, const iTargetFactory& tfactory);

void json_parse (OptRulesT& rules,
	std::istream& json_in, const iTargetFactory& tfactory);

marsh::iObject* parse (const query::Attribute& pba, const opt::GraphInfo& graphinfo);

}

#endif // OPT_PARSE_HPP
