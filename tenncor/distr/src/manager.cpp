#include "tenncor/distr/manager.hpp"

#ifdef DISTR_MANAGER_HPP

namespace distr
{

const std::string distmgr_key = "DistrManager";

void set_distrmgr (iDistrMgrptrT mgr, global::CfgMapptrT ctx)
{
	if (nullptr == ctx)
	{
		return;
	}
	ctx->rm_entry(distmgr_key);
	if (nullptr != mgr)
	{
		ctx->template add_entry<iDistrMgrptrT>(
			distmgr_key, [mgr]{ return new iDistrMgrptrT(mgr); });
	}
}

iDistrManager* get_distrmgr (const global::CfgMapptrT& ctx)
{
	if (nullptr == ctx)
	{
		return nullptr;
	}
	if (auto ptr = static_cast<iDistrMgrptrT*>(
		ctx->get_obj(distmgr_key)))
	{
		return ptr->get();
	}
	return nullptr;
}

}

#endif
