
#ifndef OPT_RULE_HPP
#define OPT_RULE_HPP

#include "opt/target.hpp"

namespace opt
{

struct OptRule final
{
	google::protobuf::RepeatedPtrField<query::Node> match_srcs_;

	TargptrT target_;
};

using OptRulesT = std::vector<OptRule>;

}

#endif // OPT_RULE_HPP
