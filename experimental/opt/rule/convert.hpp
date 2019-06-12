#include "tag/group.hpp"

#include "experimental/opt/rule/writer.hpp"

#ifndef OPT_RULE_CONVERT_HPP
#define OPT_RULE_CONVERT_HPP

namespace opt
{

namespace rule
{

struct TensBuilder
{
	virtual ~TensBuilder (void) = 0;

	virtual ade::TensptrT build (Report& report) = 0;
};

using BuilderT = std::unique_ptr<TensBuilder>;

struct Conversion final
{
	Conversion (std::string label,
		WriterptrT reporter, BuilderT builder) :
		label_(label), reporter_(reporter),
		builder_(std::move(builder)) {}

	bool valid (void) const
	{
		return nullptr != reporter_ && nullptr != builder_;
	}

	ade::TensptrT convert (tag::SubgraphsT& subs, ade::TensptrT root) const
	{
		Report report;
		reporter_->write(report, subs, root.get());
		if (report.is_success())
		{
			// match found, convert
			logs::debugf("applying conversion %s", label_.c_str());
			return builder_->build(report);
		}
		return nullptr;
	}

	std::string label_;

	WriterptrT reporter_;

	BuilderT builder_;
};

using ConversionsT = std::vector<Conversion>;

}

}

#endif // OPT_RULE_CONVERT_HPP
