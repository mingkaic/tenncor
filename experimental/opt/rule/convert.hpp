#include "experimental/opt/rule/builder.hpp"

#ifndef OPT_RULE_CONVERT_HPP
#define OPT_RULE_CONVERT_HPP

namespace opt
{

namespace rule
{

struct Conversion final
{
	Conversion (WriterptrT writer, BuilderptrT builder) :
		writer_(writer), builder_(builder) {}

	bool valid (void) const
	{
		return nullptr != writer_ && nullptr != builder_;
	}

	ade::TensptrT convert (tag::SubgraphsT& subs, ade::TensptrT root) const
	{
		Report report;
		writer_->write(report, subs, root.get());
		if (report.is_success())
		{
			// match found, convert
			return builder_->build(report, subs);
		}
		return nullptr;
	}

	WriterptrT writer_;

	BuilderptrT builder_;
};

using ConversionsT = std::vector<Conversion>;

}

}

#endif // OPT_RULE_CONVERT_HPP
