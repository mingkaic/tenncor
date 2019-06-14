
#include "experimental/opt/rule/writer.hpp"

#ifndef OPT_RULE_BUILDER_HPP
#define OPT_RULE_BUILDER_HPP

namespace opt
{

namespace rule
{

struct iBuilder
{
	virtual ~iBuilder (void) = 0;

	virtual ade::TensptrT build (Report& report,
		ade::Shape outshape) = 0;
};

using BuilderptrT = std::shared_ptr<iBuilder>;

}

}

#endif // OPT_RULE_BUILDER_HPP
