
#include "tenncor/distr.hpp"

#ifdef TENNCOR_DISTR_HPP

namespace tcr
{

const std::string distmgr_key = "DistrManager";

void set_distrmgr (distr::iDistrMgrptrT mgr, global::CfgMapptrT ctx)
{
	ctx->rm_entry(distmgr_key);
	if (nullptr != mgr)
	{
		ctx->template add_entry<distr::iDistrMgrptrT>(
			distmgr_key, [mgr](){ return new distr::iDistrMgrptrT(mgr); });
		teq::set_eval(new distr::DistrEvaluator(*mgr), ctx);
	}
	else
	{
		teq::set_eval(nullptr, ctx);
	}
}

distr::iDistrManager* get_distrmgr (global::CfgMapptrT ctx)
{
	if (auto ptr = static_cast<distr::iDistrMgrptrT*>(
		ctx->get_obj(distmgr_key)))
	{
		return ptr->get();
	}
	return nullptr;
}

}

#endif
