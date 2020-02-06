
#ifndef EXPERIMENTAL_OPT_PARSE_HPP
#define EXPERIMENTAL_OPT_PARSE_HPP

#include <google/protobuf/util/json_util.h>

#include "experimental/opt/target.hpp"
#include "experimental/opt/rule.hpp"

namespace opt
{

void parse_optimization (OptRulesT& rules,
	const Optimization& pb_opt, const iTargetFactory& tfactory);

void json_parse (OptRulesT& rules,
	std::istream& json_in, const iTargetFactory& tfactory);

marsh::iObject* parse (const query::Attribute& pba, const opt::GraphInfo& graphinfo);

}

#endif // EXPERIMENTAL_OPT_PARSE_HPP
