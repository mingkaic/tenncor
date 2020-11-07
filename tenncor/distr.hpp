
#ifndef TENNCOR_DISTR_HPP
#define TENNCOR_DISTR_HPP

#include "tenncor/distr/distr.hpp"
#include "tenncor/eteq/opsvc/opsvc.hpp"
#include "tenncor/hone/hosvc/hosvc.hpp"
#include "tenncor/serial/oxsvc/oxsvc.hpp"

namespace tcr
{

struct DistrEvaluator final : public teq::iEvaluator
{
	DistrEvaluator (distr::iDistrManager& mgr) : svc_(distr::get_opsvc(mgr)) {}

	/// Implementation of iEvaluator
	void evaluate (
		teq::iDevice& device,
		const teq::TensSetT& targets,
		const teq::TensSetT& ignored = {}) override
	{
		svc_.evaluate(device, targets, ignored);
	}

	distr::op::DistrOpService& svc_;
};

/// Make and return DistrManager and set it to context
distr::iDistrMgrptrT ctxualize_distrmgr (
	distr::ConsulptrT consul, size_t port,
	const std::string& alias = "",
	std::vector<distr::RegisterSvcF> regs = {
		distr::register_iosvc,
		distr::register_opsvc,
		distr::register_oxsvc,
	},
	const std::string& svc_name = distr::default_service,
	global::CfgMapptrT ctx = global::context());

void set_distrmgr (distr::iDistrMgrptrT mgr,
	global::CfgMapptrT ctx = global::context());

distr::iDistrManager* get_distrmgr (
	const global::CfgMapptrT& ctx = global::context());

std::string expose_node (const eteq::ETensor& etens);

std::string try_lookup_id (error::ErrptrT& err, eteq::ETensor etens);

std::string lookup_id (eteq::ETensor etens);

eteq::ETensor try_lookup_node (
	error::ErrptrT& err, const std::string& id,
	const global::CfgMapptrT& ctx = global::context());

eteq::ETensor lookup_node (const std::string& id,
	const global::CfgMapptrT& ctx = global::context());

}

#endif // TENNCOR_DISTR_HPP
