
#include "tenncor/distr.hpp"

#ifdef TENNCOR_DISTR_HPP

namespace tcr
{

const std::string distmgr_key = "DistrManager";

distr::DistrMgrptrT ctxualize_distrmgr (
	distr::ConsulptrT consul, size_t port,
	const std::string& alias, std::vector<RegisterSvcF> regs,
	const std::string& svc_name, global::CfgMapptrT ctx)
{
	auto consulsvc = distr::make_consul(consul, port, svc_name, alias);
	distr::PeerServiceConfig cfg(consulsvc, egrpc::ClientConfig(
			std::chrono::milliseconds(5000),
			std::chrono::milliseconds(10000),
			5
		));
	estd::ConfigMap<> svcs;
	for (auto& reg : regs)
	{
		assert(nullptr == reg(svcs, cfg));
	}
	auto mgr = std::make_shared<distr::DistrManager>(
		distr::ConsulSvcptrT(consulsvc), svcs);
	set_distrmgr(mgr, ctx);
	return mgr;
}

void set_distrmgr (distr::iDistrMgrptrT mgr, global::CfgMapptrT ctx)
{
	if (nullptr == ctx)
	{
		return;
	}
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

distr::iDistrManager* get_distrmgr (const global::CfgMapptrT& ctx)
{
	if (nullptr == ctx)
	{
		return nullptr;
	}
	if (auto ptr = static_cast<distr::iDistrMgrptrT*>(
		ctx->get_obj(distmgr_key)))
	{
		return ptr->get();
	}
	return nullptr;
}

}

#endif
