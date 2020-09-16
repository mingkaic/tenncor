
#include "tenncor/distr.hpp"

#ifdef TENNCOR_DISTR_HPP

namespace tcr
{

distr::DistrMgrptrT ctxualize_distrmgr (
	distr::ConsulptrT consul, size_t port,
	const std::string& alias, std::vector<distr::RegisterSvcF> regs,
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
	::tcr::set_distrmgr(mgr, ctx);
	return mgr;
}

void set_distrmgr (distr::iDistrMgrptrT mgr, global::CfgMapptrT ctx)
{
	distr::set_distrmgr(mgr, ctx);
	if (nullptr != mgr)
	{
		teq::set_eval(new DistrEvaluator(*mgr), ctx);
	}
	else
	{
		teq::set_eval(nullptr, ctx);
	}
}

distr::iDistrManager* get_distrmgr (const global::CfgMapptrT& ctx)
{
	return distr::get_distrmgr(ctx);
}

std::string expose_node (const eteq::ETensor& etens)
{
	auto mgr = get_distrmgr(etens.get_context());
	return distr::get_iosvc(*mgr).expose_node(etens);
}

std::string try_lookup_id (error::ErrptrT& err, eteq::ETensor etens)
{
	auto mgr = get_distrmgr(etens.get_context());
	if (nullptr == mgr)
	{
		err = error::error(
			"can only find reference ids using DistrManager");
		return "";
	}
	auto opt_id = distr::get_iosvc(*mgr).lookup_id(etens.get());
	if (false == bool(opt_id))
	{
		err = error::errorf("failed to find tensor %s",
			etens->to_string().c_str());
		return "";
	}
	return *opt_id;
}

std::string lookup_id (eteq::ETensor etens)
{
	error::ErrptrT err = nullptr;
	auto out = try_lookup_id(err, etens);
	if (nullptr != err)
	{
		global::fatal(err->to_string());
	}
	return out;
}

eteq::ETensor try_lookup_node (
	error::ErrptrT& err, const std::string& id,
	const global::CfgMapptrT& ctx)
{
	auto mgr = get_distrmgr(ctx);
	if (nullptr == mgr)
	{
		err = error::error(
			"can only find references using DistrManager");
		return eteq::ETensor();
	}
	return eteq::ETensor(distr::get_iosvc(*mgr).lookup_node(err, id), ctx);
}

eteq::ETensor lookup_node (const std::string& id,
	const global::CfgMapptrT& ctx)
{
	error::ErrptrT err = nullptr;
	auto out = try_lookup_node(err, id, ctx);
	if (nullptr != err)
	{
		global::fatal(err->to_string());
	}
	return out;
}

eteq::ETensor localize (const eteq::ETensor& root,
	const eteq::ETensorsT& stop,
	global::CfgMapptrT ctx)
{
	return root;
}

}

#endif
