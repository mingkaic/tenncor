#include "experimental/opt/rule/rule.hpp"

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
		NodeptrT reporter, BuilderT builder) :
		label_(label), reporter_(reporter),
		builder_(std::move(builder)) {}

	bool valid (void) const
	{
		return nullptr != reporter_ && nullptr != builder_;
	}

	ade::TensptrT convert (ade::TensptrT root) const
	{
		Report report;
        reporter_->get_report(report, root.get());
		if (report.success_)
		{
			// match found, convert
			logs::debugf("applying conversion %s", label_.c_str());
			root = builder_->build(report);
		}
		return root;
	}

	std::string label_;

	NodeptrT reporter_;

	BuilderT builder_;
};

using ConversionsT = std::vector<Conversion>;

}

}
