
#include "tenncor/distr.hpp"

#ifdef TENNCOR_DISTR_HPP

namespace tcr
{

void set_distmgr (distr::iDistMgrptrT mgr, eigen::CtxptrT ctx)
{
	ctx->owners_.erase(distmgr_key);
	if (nullptr != mgr)
	{
		ctx->owners_.insert(std::pair<std::string,eigen::OwnerptrT>{
			distmgr_key, std::make_unique<distr::ManagerOwner>(mgr)});
	}
	ctx->eval_ = std::make_shared<distr::DistEvaluator>(mgr.get());
}

}

#endif
