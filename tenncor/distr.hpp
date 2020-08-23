
#ifndef TENNCOR_DISTR_HPP
#define TENNCOR_DISTR_HPP

#include "tenncor/distrib/distrib.hpp"
#include "tenncor/eteq/opsvc/service.hpp"

namespace distr
{

struct DistrEvaluator final : public teq::iEvaluator
{
	DistrEvaluator (iDistrManager& mgr) : svc_(distr::get_opsvc(mgr)) {}

	/// Implementation of iEvaluator
	void evaluate (
		teq::iDevice& device,
		const teq::TensSetT& targets,
		const teq::TensSetT& ignored = {}) override
	{
		svc_.evaluate(device, targets, ignored);
	}

	DistrOpService& svc_;
};

}

namespace tcr
{

using RegisterSvcF = std::function<error::ErrptrT(\
	estd::ConfigMap<>&,const distr::PeerServiceConfig&)>;

/// Make and return DistrManager and set it to context
distr::DistrMgrptrT ctxualize_distrmgr (
	distr::ConsulptrT consul, size_t port,
	const std::string& alias = "",
	std::vector<RegisterSvcF> regs = {
		distr::register_iosvc,
		distr::register_opsvc
	},
	const std::string& svc_name = distr::default_service,
	global::CfgMapptrT ctx = global::context());

void set_distrmgr (distr::iDistrMgrptrT mgr,
	global::CfgMapptrT ctx = global::context());

distr::iDistrManager* get_distrmgr (
	const global::CfgMapptrT& ctx = global::context());

template <typename T>
std::string expose_node (const eteq::ETensor<T>& etens)
{
	auto mgr = get_distrmgr(etens.get_context());
	return distr::get_iosvc(*mgr).expose_node(etens);
}

template <typename T>
std::string try_lookup_id (error::ErrptrT& err, eteq::ETensor<T> etens)
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

template <typename T>
std::string lookup_id (eteq::ETensor<T> etens)
{
	error::ErrptrT err = nullptr;
	auto out = try_lookup_id(err, etens);
	if (nullptr != err)
	{
		global::fatal(err->to_string());
	}
	return out;
}

template <typename T>
eteq::ETensor<T> try_lookup_node (
	error::ErrptrT& err, const std::string& id,
	const global::CfgMapptrT& ctx = global::context())
{
	auto mgr = get_distrmgr(ctx);
	if (nullptr == mgr)
	{
		err = error::error(
			"can only find references using DistrManager");
		return eteq::ETensor<T>();
	}
	return eteq::ETensor<T>(distr::get_iosvc(*mgr).lookup_node(err, id), ctx);
}

template <typename T>
eteq::ETensor<T> lookup_node (const std::string& id,
	const global::CfgMapptrT& ctx = global::context())
{
	error::ErrptrT err = nullptr;
	auto out = try_lookup_node<T>(err, id, ctx);
	if (nullptr != err)
	{
		global::fatal(err->to_string());
	}
	return out;
}

template <typename T>
eteq::ETensor<T> localize (const eteq::ETensor<T>& root,
	const eteq::ETensorsT<T>& stop = {},
	global::CfgMapptrT ctx = global::context())
{
	return root;
}

}

#endif // TENNCOR_DISTR_HPP
